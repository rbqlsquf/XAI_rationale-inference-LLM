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
from dataclasses import dataclass

import os


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
        max_sent = encoder_outputs.size(1)
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
        tmp_result = []
        tmp_hidden = []
        tmp_attention_mask = []
        tmp_attn_outputs = []

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
            # evidence_scores_sum : [batch*topk] -> [batch *topk , topk]
            evidence_scores_sum = evidence_scores.unsqueeze(1).repeat(1, self.topk)
            # log_scores : [batch*topk, topk]
            log_scores = -torch.log(scores) + evidence_scores_sum
            l = log_scores.view(-1, self.topk * self.topk).tolist()
            index_and_scores = [sorted(enumerate(e), key=lambda x: x[1], reverse=False) for e in l]

            tmp_evidence_scores = []
            refine_evidence_sentences = []
            refine_attention_scores = []
            evidence_sentences = []
            for batch_id, index_and_score in enumerate(index_and_scores):
                tmp_evidence_scores.append([])
                refine_attention_scores.append([])
                evidence_sentences.append([])
                tmp_result.append([])
                tmp_hidden.append([])
                tmp_attention_mask.append([])
                tmp_attn_outputs.append([])
                for sample_id, sorted_node in enumerate(index_and_score[: self.topk]):
                    s, r = int(sorted_node[0] / self.topk), sorted_node[0] % self.topk
                    s = s + batch_id * self.topk
                    tmp_evidence_scores[-1].append(log_scores[s][r])
                    tmp_result[-1].append(result[s])
                    tmp_hidden[-1].append(hidden[0, s])
                    refine_evidence_sentences.append(evidence_sentence_index[s] + [sentences[s][r].item()])
                    refine_attention_scores[-1].append(
                        torch.cat([attention_scores[:, s, :, :], attn_outputs[s, :, :].unsqueeze(0)], 0)
                    )
                    evidence_sentences[-1].append(sentences[s][r])
                    tmp_attention_mask[-1].append(attention_mask[s])
                    tmp_attn_outputs[-1].append(attn_outputs[s])

                tmp_evidence_scores[-1] = torch.stack(tmp_evidence_scores[-1])
                refine_attention_scores[-1] = torch.stack(refine_attention_scores[-1])
                tmp_result[-1] = torch.stack(tmp_result[-1])
                tmp_hidden[-1] = torch.stack(tmp_hidden[-1])
                evidence_sentences[-1] = torch.stack(evidence_sentences[-1])
                tmp_attention_mask[-1] = torch.stack(tmp_attention_mask[-1])
                tmp_attn_outputs[-1] = torch.stack(tmp_attn_outputs[-1])

            evidence_scores = torch.stack(tmp_evidence_scores).view(
                -1,
            )
            attention_scores = (
                torch.stack(refine_attention_scores, 0).view(batch_size, -1, 1, max_sent).transpose(0, 1)
            )
            result = torch.stack(tmp_result, 0).view(-1, 1, self.hidden_size)
            hidden = torch.stack(tmp_hidden, 0).view(-1, self.hidden_size).unsqueeze(0)
            evidence_sentence_index = refine_evidence_sentences
            evidence_sentences = torch.stack(evidence_sentences, 0).view(
                -1,
            )
            attention_mask = torch.stack(tmp_attention_mask, 0).view(batch_size, 1, -1)
            attn_outputs = torch.stack(tmp_attn_outputs, 0).view(batch_size, 1, -1)

        else:
            evidence_scores = -torch.log(scores)[: batch_size // self.topk].reshape(-1)
            evidence_sentences = sentences[: batch_size // self.topk].reshape(-1)
            evidence_sentence_index = []

            # evidence_sentences : [batch1-1, batch1-2, batch1-3 ..., batch 4-3, batch 4-4]
            for item in evidence_sentences:
                evidence_sentence_index.append([item.item()])

            # attention_scores : [batch*topk, 1, max_sent] -> [1, batch*topk, 1, max_sent]
            attention_scores = attn_outputs.unsqueeze(0)

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
        self.evidence = None
        self.beam_size = config.beam_size
        self.gru = BeamSearchAttentionDecoder(config.hidden_size, self.max_sent, self.beam_size)
        self.max_dec_len = config.max_dec_len
        self.hidden_size = config.hidden_size

    def save_pn_model(self, model_path):
        torch.save(self.gru.state_dict(), os.path.join(model_path, "model.pt"))

    def load_pn_model(self, model_path):
        self.gru.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))

    # def generate(self, input_ids, **kwargs):
    #     # 새로운 인자 처리 예시
    #     sent_masks = kwargs.get("sent_masks", None)  # 추가 인자 예시
    #     # 필요한 로직을 여기에 추가
    #     # 예를 들어, new_arg를 사용할 수 있도록 모델 forward 함수에 추가
    #     outputs = super().generate(input_ids, **kwargs)
    #     return outputs

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
        **kwargs,
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
        tokenizer.padding_side = "left"

        # [batch, max_length, hidden]
        hidden_states = outputs[0]
        evidence_sentences = None
        mm = None
        attention_scores = None
        batch_size = input_ids.size(0)
        Flag = False  # 학습하는 과정에서 loss 출력할지 말지를 고민(학습단에서 model에 2번 진입하기 때문)

        ##############################################################################
        #                           입력부에 대한 pointer network 계산
        ##############################################################################
        if self.evidence is None:
            Flag = True
            # positions [batch, 3] -> 첫번째는 system, user, assistant
            # 위에 내용에 대해서는 target_value 즉, assistant 시작하는 부분을 찾기 위한 함수였음

            # sentence_masks : [batch, max_sent, max_length]
            sentence_masks = F.one_hot(sent_masks).transpose(1, 2).float()
            max_sent = sentence_masks.size(1)
            # div_term : [batch, max_sent, 1] : 문장들이 있는 곳에는 숫자값, 없는 곳에는 작은 소수
            div_term = torch.sum(sentence_masks, dim=-1, keepdim=True)
            div_term = div_term.masked_fill(div_term == 0, 1e-10)
            # sentence_representation : [batch, max_sent, max_length] * [batch, max_length, hidden]  -> [batch, max_sent, hidden]
            sentence_representation = sentence_masks.bmm(hidden_states)
            sentence_representation = sentence_representation / div_term

            # sentence_representation : [batch, max_sent, hidden]
            sentence_representation = self.dropout(sentence_representation)
            # 문장 없는 곳에 1, 있는 곳에 0
            sent_attention_masks = (
                div_term.masked_fill(div_term != 1e-10, 0).masked_fill(div_term == 1e-10, 1).squeeze(dim=2).bool()
            )

            sent_attention_masks = sent_attention_masks.float()

            # 질문 부분을 무시하기 위함
            sent_attention_masks[:, 0] = 1

            # 신경써야할 부분 -> 문장들이 있는 경우(이 때 질문 instruction 부분 제외)
            # mm : [batch, max_sent]
            mm = 1 - sent_attention_masks
            mm = mm.unsqueeze(1).expand(-1, self.max_dec_len, -1)
            # 이제 진짜 필요없는 부분에 대해서 엄청 큰 음수값을 넣어줌
            # sent_attention_masks : [batch, 1, max_sent] -> 확률값을 조정하기 위함으로 필요없는 문장들이 pointer nework에서 나오지 않게 하기 위함
            # sent_attention_masks : [batch, 1, max_sent] -> 확률값을 조정하기 위함으로 필요없는 문장들이 pointer nework에서 나오지 않게 하기 위함
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
            #               im_start 위치를 찾고 decoding 하는 단계
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
                    decoder_inputs.append(hidden_states[i][-1, :])
                else:  # train
                    # 전체 입력에서 마지막에 해당하는 벡터 값을 가지고 오기 위함
                    decoder_inputs.append(hidden_states[i][positions[i][2] - 1, :])

            # decoder_inputs : [batch, hidden] -> [batch, 1, hidden]
            decoder_inputs = torch.stack(decoder_inputs, 0)
            decoder_inputs = decoder_inputs.unsqueeze(dim=1)

            #################################################
            #           입력을 topk 만큼 복제
            #################################################
            # [batch, 1, hidden] -> [batch, topk, hidden]  -> [batch *topk, 1, hidden] -> 묶음으로 repeat됨 앞에 단위를 게속 반복하는 느낌
            decoder_inputs = decoder_inputs.repeat(1, self.beam_size, 1).view(-1, 1, self.hidden_size)
            encoder_outputs = encoder_outputs.repeat(1, self.beam_size, 1, 1).view(-1, max_sent, self.hidden_size)
            sent_attention_masks = sent_attention_masks.repeat(1, self.beam_size, 1).view(-1, 1, max_sent)

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

            evidence_vector = decoder_inputs.view(-1, self.beam_size, self.hidden_size)
            evidence_sentences = torch.tensor(evidence_sentences, dtype=torch.long).cuda()
            self.evidence = evidence_vector
            attention_scores = attention_scores.squeeze(2).transpose(0, 1)

        ##############################################################################
        #                   evidence_vector 만들었음                                  #
        ##############################################################################
        # element 합으로 수정
        all_path_logits = []
        for path in range(self.beam_size):
            tmp_hidden_states = hidden_states + self.evidence[:, path, :].unsqueeze(1)
            all_path_logits.append(self.lm_head(tmp_hidden_states).float())

        loss = None
        span_loss = None
        if labels is not None and Flag == True:  # 학습하는 과정
            # Shift so that tokens < n predict n
            label_losses = []
            # logits : [batch, max_length, vocab]
            loss_fct = CrossEntropyLoss()

            for logits in all_path_logits:
                batch_loss = []
                for batch_idx in range(batch_size):
                    shift_logits = logits[
                        ..., :-1, :
                    ].contiguous()  # logits의 맨 뒤에 하나 뺀거 즉 end 토큰 빼고 나서 걔랑
                    shift_labels = labels[..., 1:].contiguous()  # label에 대한 처음부터 값이 동일해야함..
                    # Flatten the tokens

                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    batch_loss.append(loss)
                # 최종 loss 계산
                label_losses.append(batch_loss)
            label_losses = torch.tensor(label_losses)
            span_loss = label_losses.cuda()

        if not return_dict:
            output = (all_path_logits[0],) + outputs[1:]
            return (span_loss,) + output if span_loss is not None else output

        return CustomCausalLMOutput(
            loss=span_loss,  # [path, 3484]
            logits=all_path_logits[0],  # [path, 2] path에 대한 문장 생성 확률??
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            evidence_sentences=evidence_sentences,  # [batch*beam_size, dec_len]
            mask=mm,
            attention_scores=attention_scores,
            path_logits=all_path_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "sent_masks": kwargs["sent_masks"],
            }
        )
        return model_inputs


@dataclass
class CustomCausalLMOutput(CausalLMOutputWithPast):

    evidence_sentences: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None
    attention_scores: Optional[torch.LongTensor] = None
    path_logits: Optional[torch.LongTensor] = None
