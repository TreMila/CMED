CUDA_VISIBLE_DEVICES=2 python merge_peft_adapter.py \
    --model_type auto \
    --base_model path_to_base_model \
    --lora_model path_to_lora_model \
    --output_dir path_to_output_dir \
