CUDA_VISIBLE_DEVICES=3 python inference_domainmodels.py \
    --model_name qwen \
    --base_model path_to_base_model \
    --data_file path_to_test_file \
    --output_dir path_to_output_dir \