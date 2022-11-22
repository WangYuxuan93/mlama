# Copyright (c) Facebook, Inc. and its affiliates
# # All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import pytorch_pretrained_bert.tokenization as btok
#from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer
from transformers import (
  BertTokenizer,
  BertForMaskedLM,
  XLMRobertaTokenizer,
  XLMRobertaForMaskedLM,
  #MEAEForMaskedLM,
)

import numpy as np
from mlama.modules.base_connector import *
import torch.nn.functional as F

BERT_MASK = "[MASK]"
XLMR_MASK = "<mask>"

class XLMRoberta(Base_Connector):

    def __init__(self, args, vocab_subset = None):
        super().__init__()

        bert_model_name = args.bert_model_name
        dict_file = bert_model_name

        if args.bert_model_dir is not None:
            # load bert model from file
            bert_model_name = str(args.bert_model_dir) + "/"
            dict_file = bert_model_name+args.bert_vocab_name
            self.dict_file = dict_file
            print("loading BERT model from {}".format(bert_model_name))
        else:
            # load bert model from huggingface cache
            pass

        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True
        #print(do_lower_case)
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(dict_file)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        # ... to get prediction/generation
        self.masked_xlmr_model = XLMRobertaForMaskedLM.from_pretrained(bert_model_name)

        self.masked_xlmr_model.eval()

        self.pad_id = self.inverse_vocab[BERT_PAD]

        self.unk_index = self.inverse_vocab[BERT_UNK]

    def _tokenize(self, string):
        str = string.replace(BERT_MASK, XLMR_MASK)
        tokenized_text = self.tokenizer.tokenize(string)
        return tokenized_text

    def get_id(self, string):
        #tokenized_text = self.tokenizer.tokenize(string)
        tokenized_text = self._tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)
        # print(final_tokens_tensor)
        # print(final_segments_tensor)
        # print(final_attention_mask)
        # print(final_tokens_tensor.shape)
        # print(final_segments_tensor.shape)
        # print(final_attention_mask.shape)
        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")
        
        #first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_tokenized_sentence = self._tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(ROBERTA_END_SENTENCE)
        first_segment_id.append(0)

        if len(sentences)>1 :
            #second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_tokenized_sentece = self._tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(ROBERTA_END_SENTENCE)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0,ROBERTA_START_SENTENCE)
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == XLMR_MASK:
                masked_indices.append(i)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_xlmr_model.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        #print("see")
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_xlmr_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            log_probs = F.log_softmax(logits, dim=-1).cpu()
        #print(logits.shape)
        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))

        return log_probs, token_ids_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_encoder_layers, _ = self.masked_xlmr_model.roberta(
                tokens_tensor.to(self._model_device),
                segments_tensor.to(self._model_device))

        all_encoder_layers = [layer.cpu() for layer in all_encoder_layers]

        sentence_lengths = [len(x) for x in tokenized_text_list]

        # all_encoder_layers: a list of the full sequences of encoded-hidden-states at the end
        # of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
        # encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size]
        return all_encoder_layers, sentence_lengths, tokenized_text_list
