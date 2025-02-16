model_name_or_path=""   # LLM底座模型路径，或者是huggingface hub上的模型名称
output_dir="/root/autodl-tmp/output"  # 填入用来存储模型的路径
dataset_name="my_dataset" # dataset_info配置的数据集名称
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


deepspeed --include=localhost:0,1,2 --master_port $MASTER_PORT src/train_bash.py \
    --deepspeed deepspeed.json \
    --do_train \
    --stage sft \
    --dataset $dataset_name \
    --finetuning_type full \
    --model_name_or_path $model_name_or_path \
    --cutoff_len 1700 \
    --template qwen \
    --output_dir $output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1987 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --save_only_model True \
    --fp16
