CUDA_VISIBLE_DEVICES=2 python inference_openmodels.py \
    --model_type_list qwen,internlm,yi,deepseek \
    --model_version_list v1,v2,v3 \
    --data_file path_to_test_file \
    --output_dir path_to_output_dir \
    --temperature 0.5 \