"""测试集评测脚本

用法:
    python eval.py                          # 评测基座模型
    python eval.py --checkpoint output/step_20  # 评测训练后的 checkpoint
    python eval.py --num_samples 8          # 每题采样 8 次
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel


def load_model(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    checkpoint_path: Optional[str] = None,
):
    """加载模型。如果指定 checkpoint，会在 4-bit base 上加载 LoRA adapter。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map=device,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    model.eval()

    if checkpoint_path is not None:
        print(f"加载 LoRA checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        model.eval()

    return model, tokenizer, device


system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


def extract_answer(completion: str) -> Optional[str]:
    """从 completion 中提取 <answer> 标签内的答案。"""
    match = re.search(r"<answer>(.*?)</answer>", completion, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def evaluate_one(
    model, tokenizer, question: str, oracle_answer: str, num_samples: int
) -> dict:
    """对单道题采样 num_samples 次，返回评测结果。"""
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [chat_prompt] * num_samples,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                max_length=512,
                pad_token_id=tokenizer.eos_token_id,
            ),
        )

    prompt_len = model_inputs["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        output_ids[:, prompt_len:], skip_special_tokens=True
    )

    exact_matches = 0
    partial_matches = 0
    no_answer = 0
    extracted_answers = []

    for completion in completions:
        extracted = extract_answer(completion)
        extracted_answers.append(extracted)
        if extracted is None:
            no_answer += 1
        elif extracted == oracle_answer:
            exact_matches += 1
            partial_matches += 1
        elif oracle_answer in extracted:
            partial_matches += 1

    return {
        "question": question,
        "oracle": oracle_answer,
        "num_samples": num_samples,
        "exact_match": exact_matches,
        "partial_match": partial_matches,
        "no_answer": no_answer,
        "extracted_answers": extracted_answers,
        "completions": completions,
    }


def main():
    parser = argparse.ArgumentParser(description="数学题测试集评测")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="LoRA checkpoint 路径"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/math_test.jsonl",
        help="测试集 JSONL 文件",
    )
    parser.add_argument(
        "--num_samples", type=int, default=4, help="每题采样次数"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="打印每题详细结果"
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(checkpoint_path=args.checkpoint)

    test_file = Path(args.test_file)
    test_problems = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            test_problems.append(json.loads(line))

    print(f"\n测试集: {args.test_file} ({len(test_problems)} 题)")
    print(f"每题采样: {args.num_samples} 次")
    print(f"总推理次数: {len(test_problems) * args.num_samples}")
    print(f"{'='*60}\n")

    total_exact = 0
    total_partial = 0
    total_no_answer = 0
    total_samples = 0

    for prob in test_problems:
        result = evaluate_one(
            model, tokenizer, prob["question"], prob["answer"], args.num_samples
        )
        total_exact += result["exact_match"]
        total_partial += result["partial_match"]
        total_no_answer += result["no_answer"]
        total_samples += args.num_samples

        if args.verbose:
            print(f"[{prob['id']}] {prob['question']}  (答案={prob['answer']})")
            print(f"  精确匹配: {result['exact_match']}/{args.num_samples}")
            print(
                f"  包含答案: {result['partial_match']}/{args.num_samples}"
            )
            print(f"  无答案:   {result['no_answer']}/{args.num_samples}")
            for i, (ans, comp) in enumerate(
                zip(result["extracted_answers"], result["completions"])
            ):
                print(f"  sample {i}: 提取='{ans}'  |  {comp[:80]}...")
            print()

    # 汇总
    exact_rate = total_exact / total_samples * 100
    partial_rate = total_partial / total_samples * 100
    no_answer_rate = total_no_answer / total_samples * 100

    print(f"{'='*60}")
    print(f"汇总结果 ({len(test_problems)} 题, 每题 {args.num_samples} 次采样)")
    print(f"  精确匹配率 (exact@1):    {total_exact}/{total_samples} = {exact_rate:.1f}%")
    print(f"  包含答案率 (pass@1):     {total_partial}/{total_samples} = {partial_rate:.1f}%")
    print(f"  未生成答案标签率:        {total_no_answer}/{total_samples} = {no_answer_rate:.1f}%")


if __name__ == "__main__":
    main()
