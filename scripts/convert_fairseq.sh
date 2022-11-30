main=/work/codes/KEXLM/src-adbpe/converter/xlmr_for_mlm_converter.py
#input=/work/experiments/mlm/saves/xlm-roberta-v0/checkpoint_0_1000000.pt
#output=/work/experiments/mlm/models/xlm-roberta-v0-1000000
input=/work/ptm/tf_from_fs/model.pt
output=/work/ptm/tf_from_fs/xlm-roberta-base

python $main --input $input --output $output
