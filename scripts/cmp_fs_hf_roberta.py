import transformers
from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer
import adbpe 
#from transformers import AutoTokenizer
#from transformers.models.meae import MEAEModel, MEAEForMaskedLM
from fairseq.data.dictionary import Dictionary
from fairseq.models.roberta import RobertaModel#, XLMRModel
#from adbpe.models.multilingual_entity_as_expert import MEAE
import torch

#line = "Hello world!"
line = "Trump lives in USA."
#line = "hellow world!"

# transformers model
print ("Input sent: ", line)
#hf_dir = "test/meae-model"
#hf_dir = "/work/ptm/transformers/xlm-roberta-base"
#hf_dir = "/work/experiments/mlm/models/xlm-roberta-v0-1800000"
hf_dir = "/work/ptm/tf_from_fs/xlm-roberta-base"
print ("##### In Transformers model #####")
print ("Model dir:", hf_dir)
tokenizer = XLMRobertaTokenizer.from_pretrained(hf_dir)
#tokenizer = AutoTokenizer.from_pretrained(hf_dir)
hf_model = XLMRobertaModel.from_pretrained(hf_dir)
#hf_model = MEAEModel.from_pretrained(hf_dir)
#hf_model = MEAEForMaskedLM.from_pretrained(hf_dir)
hf_model.eval()

toks = tokenizer.tokenize(line)
print ("tokenized tokens:", toks)
#ids = tokenizer.encode(toks)
#print (ids)
ids2 = tokenizer(line, return_tensor="pt")
print ("ids:", ids2)
ids2['input_ids'] = torch.LongTensor(ids2['input_ids']).unsqueeze(0)
ids2['attention_mask'] = torch.LongTensor(ids2['attention_mask']).unsqueeze(0)
hf_feat = hf_model(**ids2)
print ("features:\n", hf_feat.last_hidden_state)


# fairseq model
#fs_dir = "test/fairseq"
#fs_dir = "/work/ptm/fairseq/xlm-roberta-base"
#fs_dir = "/work/experiments/mlm/saves/xlm-roberta-1800000"
fs_dir = "/work/ptm/tf_from_fs"
#fs_model = MEAE.from_pretrained(fs_dir)
#fs_model = XLMRModel.from_pretrained(fs_dir)
fs_model = RobertaModel.from_pretrained(fs_dir, bpe="sentencepiece")
#print (fs_model)

print ("##### In Fairseq model #####")
print ("Model dir:", fs_dir) 
fs_ids = fs_model.encode(line)
print ("ids:", fs_ids)
fs_model.model.eval()
fs_feat = fs_model.extract_features(fs_ids)
print ("features:\n", fs_feat)

#dic = Dictionary.load("/work/ptm/fairseq/xlm-roberta-base/dict.txt")
#fs_ids = [dic.bos()]+[dic.index(t) for t in toks]+[dic.eos()]
#print (fs_ids)


