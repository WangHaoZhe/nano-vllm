import os
import time
import argparse
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

app = FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 256


llm = None
tokenizer = None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # only add this for qwen3
    )

    sampling_params = SamplingParams(max_tokens=request.max_tokens)

    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0]["text"]
    # print(f"Generated chat completion: {generated_text}")

    response_id = f"chatcmpl-{int(time.time())}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(generated_text),
            "total_tokens": len(prompt) + len(generated_text),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="../Llama-3-8B-Instruct-QServe")
    parser.add_argument("--model", type=str, default="../Qwen3-8B-QServe")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    param_path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(param_path)

    llm = LLM(param_path, enforce_eager=True, tensor_parallel_size=1)

    uvicorn.run(app, host=args.host, port=args.port)
