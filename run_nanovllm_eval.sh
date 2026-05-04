# MODEL_NAME="mit-han-lab/Llama-3-8B-Instruct-QServe"
MODEL_NAME="AlbertWangSJTU/Qwen3-8B-QServe"
MODEL_PATH="http://localhost:8123/v1/chat/completions"
TOKENIZER_PATH=${MODEL_PATH}

lm_eval     --model local-chat-completions \
    --model_args model=${MODEL_NAME},base_url=${MODEL_PATH},num_concurrent=1,max_retries=3,max_length=1000     \
    --tasks mmlu_generative     \
    --output_path ./eval_out    \
    --log_samples   \
    --cache_requests true    \
    --apply_chat_template