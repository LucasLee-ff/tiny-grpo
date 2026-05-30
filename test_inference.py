"""Qwen2.5-0.5B-Instruct 推理测试脚本"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"设备: {device}")
    print(f"模型: {model_name}")
    print("加载中...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=True,
    )
    model.eval()

    # Qwen2.5 的 ChatML 格式
    chat_messages = [
        {"role": "system", "content": "你是一个数学助手。请用中文回答，先展示推理过程，再给出最终答案。"},
        {"role": "user", "content": "小明有5个苹果，吃了2个，又买了3个，现在有几个苹果？"},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n渲染后的 prompt (前 200 字符):\n{chat_prompt[:200]}...\n")

    model_inputs = tokenizer(
        [chat_prompt], return_tensors="pt", padding=True, padding_side="left"
    ).to(device)

    print("生成中...\n")

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                max_new_tokens=256,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )

    completion = tokenizer.batch_decode(
        output_ids[:, model_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

    print("模型回答:")
    print(completion)

    # 显存占用
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"\n显存占用: {vram:.2f} GB")


if __name__ == "__main__":
    main()
