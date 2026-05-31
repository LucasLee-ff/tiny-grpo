import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from collections.abc import Callable
import csv
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[PeftModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
        quantization_config=bnb_config,
        local_files_only=True,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:

    model.eval()

    input_len = 0
    try:
        # 1. format prompt
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(
            [chat_prompt],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            return_attention_mask=True,
        ).to("cuda")

        input_len = model_inputs["input_ids"].shape[1]

        # duplicate prompt num_rollouts times
        model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
            num_rollouts, 1
        )
        input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
        model_inputs["input_ids"] = input_ids

        # 2. sample completions
        pad_token_id = tokenizer.eos_token_id
        generation_config = GenerationConfig(
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_length,
            pad_token_id=pad_token_id,
        )
        sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
        completions = tokenizer.batch_decode(
            sequence_ids[:, input_len:], skip_special_tokens=True
        )
        print(f">>> sample completion: \n{completions[0][:200]}")

        action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        action_mask[:, input_len:] = True
        action_mask[sequence_ids == pad_token_id] = False
        action_mask = action_mask[:, 1:]

        # 3. determine rewards
        returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
        for i, completion in enumerate(completions):
            answer_match = re.search(
                r"<answer>(.*?)</answer>",
                completion,
                flags=re.DOTALL,
            )

            answer = answer_match.group(1) if answer_match else None
            reward = 0
            if answer is not None:
                if answer == oracle_answer:
                    reward = 1.0
                elif oracle_answer in answer:
                    reward = 0.5
                else:
                    reward = 0.01

            returns[i] = reward

        return sequence_ids, returns.to(sequence_ids.device), action_mask, completions

    except Exception as e:
        print(f"[rollout ERROR] task: {task!r}")
        print(f"[rollout ERROR] oracle_answer: {oracle_answer!r}")
        print(f"[rollout ERROR] input_len: {input_len}")
        print(f"[rollout ERROR] num_rollouts: {num_rollouts}")
        print(f"[rollout ERROR] exception: {type(e).__name__}: {e}")
        print(f"[rollout ERROR] 跳过该样本，继续下一个...")
        return None


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: PeftModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 4
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 4
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 128
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    model, tokenizer = load_model(model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    metrics_path = Path("metrics.csv")
    metrics_file = metrics_path.open("w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "returns", "loss", "kl", "grad_norm"])

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                result = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                if result is None:
                    continue
                sequence_ids, returns, action_mask, completions = result

                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                with model.disable_adapter():
                    log_probs_ref = sequences_log_probs(
                        model=model,
                        sequence_ids=sequence_ids,
                        attention_mask=attention_mask,
                    )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))
                print(
                    f">>> rollout\n q='{q}', a='{a}', returns={returns.sum().item():.2f}, replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                )
                print("-" * 80)

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        train_losses = []
        train_kls = []
        train_grad_norms = []

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward, loss={loss}")
                    print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")
                wandb.log({"kl": kl, "grad_norm": grad_norm})

                optimizer.step()

                train_losses.append(loss.item())
                train_kls.append(kl.mean().item())
                train_grad_norms.append(grad_norm.item())

        step_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        step_kl = sum(train_kls) / len(train_kls) if train_kls else 0
        step_grad_norm = sum(train_grad_norms) / len(train_grad_norms) if train_grad_norms else 0

        metrics_writer.writerow([k, episode_return_sum.item(), step_loss, step_kl, step_grad_norm])
        metrics_file.flush()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")

    metrics_file.close()
    print(f"metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
