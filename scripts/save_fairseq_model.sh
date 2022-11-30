MAIN_DIR=/work/codes/KEXLM/src-adbpe/train_mlm.py
DATA_DIR=/work/wiki/mlm
MODEL_DIR=/work/ptm/fairseq
SAVE_DIR=/work/ptm/tf_from_fs
LOG_DIR=/tmp
model_path=/work/ptm/fairseq/xlm-roberta-base/model.pt
GPU=0 #,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
${MAIN_DIR} ${MODEL_DIR}/xlm-roberta-base/ \
--ddp-backend=no_c10d --distributed-no-spawn \
--task mlmv2_infinibatch_v2 --criterion masked_lm \
--arch reload_roberta_base \
--roberta-model-path $model_path \
--optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-6 \
--lr-scheduler polynomial_decay --lr 1e-4 --clip-norm 1.0 \
--batch-size 32 --max-tokens 512 --max-positions 512 \
--update-freq 1 \
--warmup-updates 10000 --total-num-update 7680000 --max-update 7680000 \
--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
--log-format simple --log-interval 100 --disable-validation \
--save-interval-updates 10000 \
--no-epoch-checkpoints \
--fp16-init-scale 128 --fp16-scale-window 128 --min-loss-scale 0.0001 \
--seed 1 \
--save-dir ${SAVE_DIR} --tensorboard-logdir ${LOG_DIR} \
--num-workers 1 \
--curriculum 100000000 \
--reset-dataloader \
--mlm_data ${DATA_DIR}/tokenized \
--mlm_data_json ${DATA_DIR}/configs/data_files.json \
--lang_prob ${DATA_DIR}/configs/lp_0.7.json \
--mask-prob 0.15 \
--random-token-prob 0.10 \
--leave-unmasked-prob 0.10 \
--fp16 \

#--pretrained-model-path ${MODEL_DIR}/xlm-roberta-base/model.pt \
#--all-gather-list-size 65536 \

#--entity_vocab_path ${DATA_DIR}/entity_vocab/mluke_entity_vocab.jsonl \
#--entity_embed_dim 256 \
#--entity-mask-prob 0.20 \
#--entity_memory_after_layers 3 \
#--use_entity_linking_loss \
#--use_entity_pred_loss \

#--use_bio_loss \
#--fix_pretrained_params \
# --arch multilingual_entity_as_expert_base \
