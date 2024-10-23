import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers import Qwen2ForCausalLM
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers import AutoTokenizer


class BeamSearchAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, num_sent, topk=1):
        super(BeamSearchAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_sent = num_sent
        self.dense1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.dense3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)
        self.topk = topk

    def forward(
        self,
        last_hidden,
        decoder_inputs,
        encoder_outputs,
        attention_scores,
        attention_mask,
        evidence_scores=None,
        evidence_sentence_index=None,
    ):
        """
        :param last_hidden: (1, batch, hidden)
        :param decoder_inputs: (batch, 1, hidden)
        :param encoder_outputs: (batch, seq_len, hidden)
        :return:
        """
        # decoder_input : [batch, hidden] # 주의!! 실제 batch보다 topk배 많음
        batch_size = decoder_inputs.size(0)
        indexes = [e for e in range(batch_size)]
        key_encoder_outputs = self.dense1(encoder_outputs)
        value_encoder_outputs = self.dense2(encoder_outputs)

        # key : (batch, seq, hidden)
        # value : (batch, seq, hidden)

        output, hidden = self.decoder(decoder_inputs, hx=last_hidden)
        # output : (batch(20), 1, hidden)
        # hidden : (1, batch(20), hidden)
        # t_encoder_outputs : (batch*topk, hidden, seq)
        t_encoder_outputs = key_encoder_outputs.transpose(1, 2)

        # attn_outputs : (batch*topk, 1, max_sent 40), attention_mask : 필요없는 부분이 -1이기 때문에 확률 이빠이 낮춰
        attn_outputs = output.bmm(t_encoder_outputs) / self.div_term + attention_mask

        # attn_alignment : [batch*topk, 1, max_sent 40]
        attn_alignment = F.softmax(attn_outputs, -1)
        context = attn_alignment.bmm(value_encoder_outputs)
        # context : (batch*topk, 1, hidden)

        hidden_states = torch.cat([context, output], -1)
        # result : [batch*topk, 1, hidden]
        result = self.dense3(hidden_states)  # context와 output을 concat한 값

        #################################################################
        #                일단 Greedy 하게 진행
        #################################################################

        top_n_logit_indices = attn_alignment.topk(k=self.topk, dim=-1, sorted=True)
        # scores : [batch*topk, 1(topk)] , sentences : [batch*topk, 1]
        scores = top_n_logit_indices.values.squeeze(1)
        sentences = top_n_logit_indices.indices.squeeze(1)
        # sentences = torch.argmax(attn_alignment, -1)
        # scores = attn_alignment[:, :, torch.argmax(attn_alignment)]
        # evidence_scores : [batch,topk]

        # 두번째 decoding step!
        if evidence_scores is not None:
            ####################################################################
            #                  evidence_scores_sum을 계산함 문장들의 log 확률 더함
            ####################################################################
            # evidence_scores_sum : [batch, topk] -> [batch , topk * topk]
            evidence_scores_sum = evidence_scores.repeat(1, self.topk)
            # log_scores : [batch, topk*topk]
            log_scores = -torch.log(scores.repeat(1, self.topk)) + evidence_scores_sum
            l = log_scores.tolist()
            index_and_scores = [sorted(enumerate(e), key=lambda x: x[1], reverse=False) for e in l]

            refine_evidence_sentences = []
            for batch_id, item in enumerate(evidence_sentences):
                # 더 추가적으로 적어줘야하니까
                refine_evidence_sentences.append(evidence_sentence_index[batch_id] + [item.item()])
            evidence_sentence_index = refine_evidence_sentences

        else:
            evidence_scores = -torch.log(scores)[: batch_size // self.topk]
            evidence_sentences = sentences[: batch_size // self.topk].reshape(-1)
            evidence_sentence_index = []

            # evidence_sentences : [batch1-1, batch1-2, batch1-3 ..., batch 4-3, batch 4-4]
            for item in evidence_sentences:
                evidence_sentence_index.append([item.item()])

            # attention_scores : [batch*topk, 1, max_sent]
            attention_scores = attn_outputs

        # 근거 문장의 확률 낮춤
        attention_mask[indexes, 0, evidence_sentences] = -1e10
        # evidence_scores : [batch, topk 여야함]
        # evidence_sentence_index : 리스트 각 batch마다 이제 근거 문장들이 들어갈 예정
        # attention_scores : [batch, 1, max_sent]

        # decoder_inputs, last_hidden, evidence_sentences, attention_scores, sent_attention_masks, evidence_scores,
        # evidence_scores : path 별 누적 점수 (beam search에서 상위 N개 뽑을때 사용)
        # attention_scores : 각 decoding step 별 문장 추출 logits
        return result, hidden, evidence_sentence_index, attention_scores, attention_mask, evidence_scores


class Qwen2ForCausalLM_pn(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)  # 부모 클래스 초기화

        # 추가 레이어와 속성
        self.dropout = nn.Dropout(0.1)
        self.max_sent = 60
        self.test = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.evidence = None
        self.beam_size = config.beam_size
        self.gru = BeamSearchAttentionDecoder(config.hidden_size, self.max_sent, self.beam_size)
        self.max_dec_len = config.max_dec_len

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sent_masks: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        new_special_tokens = {"additional_special_tokens": ["<|mrc|>", "<|summary|>"]}
        tokenizer.add_special_tokens(new_special_tokens)
        tokenizer.padding_side = "left"
        # [batch, max_length, hidden]
        hidden_states = outputs[0]

        if self.evidence is None:
            batch_size = input_ids.size(0)

            # positions [batch, 3] -> 첫번째는 system, user, assistant
            # 위에 내용에 대해서는 target_value 즉, assistant 시작하는 부분을 찾기 위한 함수였음
            # [batch, max_sent, max_length]
            sentence_masks = F.one_hot(sent_masks).transpose(1, 2).float()
            # div_term : [batch, max_sent, 1]
            div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
            div_term = div_term.masked_fill(div_term == 0, 1e-10)

            sentence_representation = sentence_masks.bmm(hidden_states)
            sentence_representation = sentence_representation / div_term

            # sentence_representation : [batch, max_sent, hidden]
            sentence_representation = self.dropout(sentence_representation)
            # 문장들이 없는 곳에 1을 넣는거임... 왜? 무시할 곳에 1을 넣음 -> 나중에 디코딩단계에서 확률을 낮춰주기 위함
            sent_attention_masks = (
                div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).squeeze(dim=2).bool()
            )

            sent_attention_masks = sent_attention_masks.float()

            # 질문 부분을 무시하기 위함
            sent_attention_masks[:, 0] = 1

            # 신경써야할 부분 -> 문장들이 있는 경우(이 때 질문 instruction 부분 제외)
            # mm : [batch, max_sent]
            mm = 1 - sent_attention_masks

            # 이제 진짜 필요없는 부분에 대해서 엄청 큰 음수값을 넣어줌
            # sent_attention_masks : [batch, 1, max_sent]
            sent_attention_masks = (
                sent_attention_masks.masked_fill(sent_attention_masks == 1, -1e10)
                .masked_fill(sent_attention_masks == 0, 0)
                .unsqueeze(1)
            )
            # 디코딩 시작
            last_hidden = None
            # encoder_outputs: [batch, max_length, hidden]
            encoder_outputs = sentence_representation

            ################################################################################
            # im_start 위치를 찾고 decoding 하는 단계
            ################################################################################
            target_value = tokenizer.encode("<|im_start|>")[0]
            mask = torch.eq(input_ids, target_value)
            max_positions = 3
            positions = torch.zeros(batch_size, max_positions, dtype=torch.long)
            for i in range(batch_size):
                pos = torch.nonzero(mask[i]).squeeze()
                pos = pos[:max_positions] if pos.numel() > max_positions else pos
                positions[i, : pos.numel()] = pos

            decoder_inputs = []
            for i in range(batch_size):
                if positions[i][2] == 0:  # first inference
                    values = hidden_states[i][positions[i][1] :]
                    decoder_inputs.append(hidden_states[i][-1, :])
                else:  # train
                    values = hidden_states[i][positions[i][1] : positions[i][2]]

                    # 전체 입력에서 마지막에 해당하는 벡터 값을 가지고 오기 위함...
                    decoder_inputs.append(hidden_states[i][positions[i][2] - 1, :])

            # decoder_inputs : [batch, hidden]
            decoder_inputs = torch.stack(decoder_inputs, 0)
            decoder_inputs = decoder_inputs.unsqueeze(dim=1)

            #################################################
            #           입력을 topk 만큼 복제
            #################################################
            # [batch, 1, hidden] -> [batch *topk, 1, hidden] -> 묶음으로 repeat됨
            decoder_inputs = decoder_inputs.repeat(self.beam_size, 1, 1)
            encoder_outputs = encoder_outputs.repeat(self.beam_size, 1, 1)
            sent_attention_masks = sent_attention_masks.repeat(self.beam_size, 1, 1)

            evidence_scores = None
            evidence_sentences = []
            attention_scores = []
            #################################################
            #               디코더 들어갈 위치               #
            #################################################
            for evidence_step in range(self.max_dec_len):  # max_dec_len : 근거 문장 수
                (
                    decoder_inputs,
                    last_hidden,
                    evidence_sentences,
                    attention_scores,
                    sent_attention_masks,
                    evidence_scores,
                ) = self.gru(
                    last_hidden,
                    decoder_inputs,
                    encoder_outputs,
                    attention_scores,
                    sent_attention_masks,
                    evidence_scores,
                    evidence_sentences,
                )

            evidence_vector = decoder_inputs
            self.evidence = evidence_vector
            evidence_sentences = torch.tensor(evidence_sentences, dtype=torch.long).cuda()

        ##############################################################################
        #                   evidence_vector 만들었음                                  #
        ##############################################################################
        # element 합으로 수정
        tmp_hidden_states = hidden_states + self.evidence
        hidden_states = tmp_hidden_states
        logits = self.lm_head(hidden_states)  # 토큰 중에 최대 값이 나오는 확률 찾기
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # logits의 맨 뒤에 하나 뺀거 즉 end 토큰 빼고 나서 걔랑
            shift_labels = labels[..., 1:].contiguous()  # label에 대한 처음부터 값이 동일해야함..
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # evidence_sentence =
            # evidence_logits =
        )
