import pickle
import os
import numpy as np

problem = []
f_out = open("./output/meae_1000k_ranked.csv", "w")
output_path = "./output/results/meae_base-mlm-10000/"
path_compare = "./output/results/mbert_base-v2/"
languages = list(os.walk(output_path))[0][1:-1][0]
dict_languages_total = {}
dict_languages_P = {}

f_out.write("lc,")
f_out.write("\ttotal_all,")
f_out.write("\tP-meae-base,")
f_out.write("\tP-mbert-v2")
f_out.write("\n")

for lang in languages:
  print(lang)
  P_all = []
  P_all_eng = []
  total_all = []
  relations = list(os.walk(output_path + lang + "/"))[0][1:-1][0]
  for relation in relations:
       if "date" in relation:
           continue
       P = 0.0
       P_eng = 0.0
       total = 0.0

       with open(output_path + lang + "/" + relation + "/" + 'result.pkl', 'rb') as f:
            data = pickle.load(f)

       with open(path_compare + lang + "/" + relation + "/" + 'result.pkl', 'rb') as f:
            data_eng = pickle.load(f)

       if len(data["list_of_results"]) >0:
           eng_dict = {}
           for d in data_eng["list_of_results"]:
               rank = 0.0
               if d['masked_topk']["rank"]==0:
                   rank = 1.0
               eng_dict[d["sample"]["uuid"]] = [rank, d["sample"]]
           for d in data["list_of_results"]:
               rank = 0.0
               if d['masked_topk']["rank"]==0:
                   rank = 1.0
               P += rank
               total += 1.0
               idx = int(d["sample"]["uuid"])
               if idx in eng_dict:
                   P_eng += eng_dict[idx][0]

           P_all.append(P/total)
           P_all_eng.append(P_eng/total)
           total_all.append(total)

  f_out.write(lang)
  f_out.write("\t")
  f_out.write("{}".format(np.sum(total_all)))
  f_out.write(",\t")
  f_out.write("{:.4f}".format(np.mean(P_all)))
  f_out.write(",\t")
  f_out.write("{:.4f}".format(np.mean(P_all_eng)))
  f_out.write("\n")
f_out.close()
