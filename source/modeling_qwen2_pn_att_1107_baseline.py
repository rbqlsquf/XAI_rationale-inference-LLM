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
from torch.nn import functional as F
import os
from modeling_qwen2_ import Qwen2Model


class BeamSearchAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, num_sent, topk=1):
        super(BeamSearchAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_sent = num_sent

        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=1)

        self.div_term = math.sqrt(hidden_size)
        self.topk = topk  # beam_size랑 동일, 인자로 첨부터 넘겨받긴함
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)

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
        key_encoder_outputs = self.key_linear(encoder_outputs)
        value_encoder_outputs = self.value_linear(encoder_outputs)
        # key_encoder_outputs = key_encoder_outputs + encoder_outputs
        # value_encoder_outputs = self.dense2(encoder_outputs)
        # value_encoder_outputs = value_encoder_outputs + encoder_outputs

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

        # hidden_states = torch.cat([context, output], -1)
        # result : [batch*topk, 1, hidden]
        result = context  # self.dense3(hidden_states)  # context와 output을 concat한 값
        # tmp_list = []
        # tmp_evi_sentence_index = attn_alignment.argmax(dim=-1)
        # for batch_idx in range(batch_size):
        #     tmp_list.append(encoder_outputs[batch_idx, tmp_evi_sentence_index[batch_idx], :].squeeze(0))

        # evi_sent_representation = torch.stack(tmp_list)
        # evi_sent_representation = torch.stack([encoder_outputs[batch_idx, evidence_sentence_index[batch_idx], :]] for batch_idx in range(batch_size))
        # result = evi_sent_representation.unsqueeze(1)

        ##################################################################
        #               beam search 계산
        ##################################################################
        tmp_result = []
        tmp_hidden = []
        tmp_attention_mask = []
        tmp_attn_outputs = []
        batch_attn_alignment = attn_alignment.view(batch_size // self.topk, -1, max_sent)
        top_n_logit_indices = []
        for real_batch in range(batch_size // self.topk):
            top_n_logit_indices.append(batch_attn_alignment[real_batch].topk(k=self.topk, dim=-1, sorted=True))
        # top_n_logit_indices = attn_alignment.topk(k=self.topk, dim=-1, sorted=True)

        # scores : [batch*topk, 1(topk)] , sentences : [batch*topk, 1]
        scores = []
        sentences = []
        for top_n_logit_index in top_n_logit_indices:
            scores.append(top_n_logit_index.values)
            sentences.append(top_n_logit_index.indices)
        # sentences = torch.argmax(attn_alignment, -1)
        # scores = attn_alignment[:, :, torch.argmax(attn_alignment)]
        # evidence_scores : [batch,topk]
        # encoder_outputs : [batch, sent_len, hidden]
        scores = torch.stack(scores).view(batch_size, -1)
        sentences = torch.stack(sentences).view(batch_size, -1)

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
            # evidence_scores = -torch.log(scores[: batch_size // self.topk]).reshape(-1)
            # evidence_sentences = sentences[: batch_size // self.topk].reshape(-1)

            # 인덱스 생성: 0, a, 2a, ...
            row_indices = torch.arange(0, sentences.size(0), step=self.topk)
            evidence_scores = -torch.log(scores[row_indices]).reshape(-1)
            evidence_sentences = sentences[row_indices].reshape(-1)
            evidence_sentence_index = []

            # evidence_sentences : [batch1-1, batch1-2, batch1-3 ..., batch 4-3, batch 4-4]
            for item in evidence_sentences:
                evidence_sentence_index.append([item.item()])

            # attention_scores : [batch*topk, 1, max_sent] -> [1, batch*topk, 1, max_sent]
            attention_scores = attn_outputs.unsqueeze(0)

        # 근거 문장의 확률 낮춤
        # attention_mask[indexes, 0, evidence_sentences] = -1e10
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
        self.linear_w1 = nn.Linear(in_features=config.hidden_size * 2, out_features=config.hidden_size)
        self.gru = None
        self.model = Qwen2Model(config)

        self.max_dec_len = config.max_dec_len
        self.hidden_size = config.hidden_size

        self.sentence_number = None

    def set_gru(self, gru):
        self.gru = gru

    def save_pn_model(self, model_path):
        state_dict = {"gru": self.gru.state_dict(), "linear_w1": self.linear_w1.state_dict()}
        torch.save(state_dict, os.path.join(model_path, "model.pt"))

    def load_pn_model(self, model_path):
        state_dict = torch.load(os.path.join(model_path, "model.pt"))
        self.gru.load_state_dict(state_dict["gru"])
        self.linear_w1.load_state_dict(state_dict["linear_w1"])

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
        gold_sp: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if self.evidence is not None:
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
                sent_masks=None,
            )
        else:
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
                sent_masks=sent_masks,
                labels=labels,
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
        ##############################################################################
        #                   evidence_vector 만들었음                                  #
        ##############################################################################
        # element 합으로 수정
        all_path_logits = []
        for path in range(self.beam_size):
            # hidden_states : (batch, max_length, hidden)
            # self.evidence : (batch, 1, hidden)

            tmp_hidden_states = hidden_states
            all_path_logits.append(self.lm_head(tmp_hidden_states).float())

        loss = None
        span_loss = None
        if labels is not None:  # 학습하는 과정
            # Shift so that tokens < n predict n
            label_losses = []

            # logits : [batch, max_length, vocab]
            loss_fct = CrossEntropyLoss()

            for logits in all_path_logits:
                for batch_idx in range(batch_size):

                    shift_logits = logits[batch_idx][
                        ..., :-1, :
                    ].contiguous()  # logits의 맨 뒤에 하나 뺀거 즉 end 토큰 빼고 나서 걔랑
                    shift_labels = labels[batch_idx][..., 1:].contiguous()  # label에 대한 처음부터 값이 동일해야함..
                    # Flatten the tokens

                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                    # 최종 loss 계산
                    label_losses.append(loss)
            label_losses = torch.stack(label_losses, 0)
            span_loss = label_losses.view(-1, batch_size)

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
