export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# model setting
MODEL_NAME=NousResearch/Llama-2-7b-hf


CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --gpu_ids 4,5,6,7 \
    --use_deepspeed \
    --deepspeed_config_file stage3_no_offloading_accelerate.conf \
    src.finetune.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --use_slow_tokenizer \
    --train_file data/train.jsonl \
    --max_seq_length  1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 1 \
    --output_dir output/adaptive_${MODEL_SIZE}/ \
    --with_tracking \
    --logging_steps 1 \
    --report_to wandb \