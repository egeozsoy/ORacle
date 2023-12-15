# Start demo with finetuned model

## Controller

python -m llava.serve.controller --host 0.0.0.0 --port 10000

## Gradio

python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload

## Worker

CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path
/home/guests/chantal_pellegrini/Oracle/LLaVA/checkpoints/llava-v1.5-7b-task-lora_4dor_qlora --model-base liuhaotian/llava-v1.5-7b

# TODO try using: liuhaotian/llava-v1.5-7b
