export CUDA_VISIBLE_DEVICES=4,5,6,7
# export WANDB_MODE=disabled
WANDB_PROJECT=llama_peft


MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# lora setting
lr=1e-4
lora_rank=8
lora_alpha=32
modules_to_save="embed_tokens"
# modules_to_save="null"
lora_dropout=0.1

# model setting
MODEL_NAME=NousResearch/Llama-2-7b-hf


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nnodes 1\
    --nproc_per_node $NUM_GPUS \
    --master_port 29500 \
    src.train_ws_torchrun.py \
    --deepspeed ds_config/stage2.conf \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --do_train \
    --do_eval \
    --train_file $dataset_dir/train.json \
    --validation_file $dataset_dir/dev.json \
    --max_seq_length  1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --output_dir output/adaptive_${MODEL_SIZE}/ \
    --with_tracking \
    --logging_steps 1 \
    --report_to wandb \


