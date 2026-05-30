# tiny-grpo 训练流程详解

## 一、入口：`main()` （train.py 第231行）

### 1.1 超参数设置

```python
seed = 42
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
```

关键超参数分三类：

| 类别 | 参数 | 默认值 | 含义 |
|---|---|---|---|
| **RL** | `group_size` | 12 | 每个 prompt 生成多少个 completion，用于组内归一化 advantage |
| **RL** | `clip_eps` | 0.2 | PPO 裁剪范围 `[0.8, 1.2]` |
| **RL** | `kl_weight` | 0.01 | KL 惩罚系数，防止策略偏离参考模型太远 |
| **训练** | `lr` | 5e-6 | Adam 学习率（只优化 LoRA 权重） |
| **训练** | `train_batch_size` | 16 | 从 replay buffer 采样的 batch 大小 |
| **训练** | `rollouts_per_step` | 32 | 每步处理多少个不同的 prompt |
| **训练** | `epochs_per_step` | 1 | 每步在 replay buffer 上训练几轮 |
| **生成** | `max_length` | 1024 | 生成最大 token 数 |
| **生成** | `temperature` | 1.0 | 采样温度 |
| **生成** | `top_p` | 1.0 | nucleus sampling |

### 1.2 模型加载

```python
model, tokenizer = load_model(model_name, device_map=device)
```

只加载**一个**模型，同时充当训练策略和参考策略。详情见下文 `load_model()`。

### 1.3 数据加载

```python
prompts = read_prompts("data/math_tasks.jsonl",
    predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
    max_rows=64 * 1024,
)
```

从 JSONL 中读取数学题，过滤条件：题目长度 < 128 字符、项数 ≤ 3、位数 ≤ 3。用 `DataLoader` 包装，每批 `rollouts_per_step`（32）个 prompt。

---

## 二、模型加载：`load_model()` （train.py 第30行）

按顺序执行四个步骤：

### Step 1: 4-bit 量化加载

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4，信息论最优
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化到 bf16
    bnb_4bit_use_double_quant=True,     # 二重量化，再省 ~10% 显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    attn_implementation="sdpa",   # PyTorch 内置，无需编译 flash-attn
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
```

模型从 ~2GB（bf16）压缩到 ~400MB（4-bit）。

### Step 2: 梯度检查点

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

用计算换显存：不保存中间激活，反向传播时重新计算。对 8GB 显卡至关重要。

### Step 3: k-bit 训练准备

```python
model = prepare_model_for_kbit_training(model)
```

将 LayerNorm 等不稳定的层转为 FP32，启用 embedding 层梯度。

### Step 4: 挂载 LoRA

```python
lora_config = LoraConfig(
    r=16, alpha=32, dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

只在 7 个线性投影层上添加低秩适配器。可训练参数 ~880 万（占总参数 0.5%），基础模型权重冻结。

---

## 三、训练循环（train.py 第286行）

每个 step 分为两大阶段：**Rollout（采集经验）** 和 **Training（策略更新）**。

### 3.1 Rollout 阶段

对 batch 中的每个 prompt（共 32 个），执行：

#### a) 生成 completions — `rollout()`（train.py 第85行）

1. **构建 prompt**：用 `tokenizer.apply_chat_template()` 将 system prompt + user question 渲染为 ChatML 格式。Qwen 的模板是：
   ```
   <|im_start|>system
   ...<|im_end|>
   <|im_start|>user
   ...<|im_end|>
   <|im_start|>assistant
   ```

2. **复制 prompt**：将 prompt 重复 `group_size`（12）次，一次 forward 生成 12 条不同的 completion。

3. **生成**：`model.generate()` 采样生成，此时 **LoRA 启用**，用的是当前策略。

4. **计算 reward**：用正则 `r"<answer>(.*?)</answer>"` 提取答案，与标准答案比对：
   - 精确匹配 → reward = 1.0
   - 答案包含标准答案 → reward = 0.5
   - 有 answer 标签但不匹配 → reward = 0.01
   - 没有 answer 标签 → reward = 0

#### b) 计算 advantage（组内归一化）

```python
advantages = (returns - returns.mean()) / (returns.std() + eps)
```

在每组 12 个 completion 内部做标准化。这意味着：同一组内，好的 completion 有正 advantage，差的 completion 有负 advantage。这是 GRPO 的核心——**不需要外部的价值函数（critic）**。

#### c) 计算 log probs（当前策略）

```python
log_probs = sequences_log_probs(model, sequence_ids, attention_mask)
```

LoRA 启用状态下的前向传播，得到当前策略对每个 token 的对数概率。

#### d) 计算 log probs（参考策略）

```python
with model.disable_adapter():
    log_probs_ref = sequences_log_probs(model, sequence_ids, attention_mask)
```

临时禁用 LoRA，用原始 Qwen 权重计算。这个参考 log prob 用于 KL 散度约束，防止策略跑偏。

#### e) 计算 KL 散度

```python
kl = approx_kl_divergence(log_probs, log_probs_ref, action_mask)
```

用的是 **k3 估计器**（来自 John Schulman 的[博客](http://joschu.net/blog/kl-approx.html)）：

```
KL ≈ exp(log_ratio) - log_ratio - 1
```

其中 `log_ratio = log_probs_ref - log_probs`（注意：是 ref 减当前）。

#### f) 存入 ReplayBuffer

每条 completion 被 `split_experience_batch()` 拆成单独的 `Experience`，存入 buffer。一个 step 会收集 32 × 12 = 384 条 experience。

### 3.2 Training 阶段

```python
for step_epoch in range(epochs_per_step):  # 默认 1 轮
    model.train()
    for exp in experience_sampler:
```

从 ReplayBuffer 随机采样 batch（16 条），每条 experience 包含采集时记录的 `action_log_probs`（旧策略的 log prob）、`advantages`、`action_mask` 等。

对每个 batch：

1. **重新计算当前 log probs**
   ```python
   log_probs = sequences_log_probs(model, exp.sequences, exp.attention_mask)
   ```
   LoRA 权重已经更新过了，所以这个 log_probs 和 experience 中记录的 `action_log_probs` **不同**。

2. **计算 GRPO loss**
   ```python
   loss, kl = objective(log_probs=log_probs, experience=exp)
   ```

3. **反向传播 + 梯度裁剪 + 参数更新**
   ```python
   loss.backward()
   grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```

### 3.3 Checkpoint

每 `checkpoint_interval`（20）步，保存 LoRA 适配器权重：

```python
model.save_pretrained(checkpoint_path / f"step_{k}")
```

只保存几 MB 的 adapter，不保存 4-bit 基础模型。

---

## 四、GRPO Loss 详解（loss.py）

```python
class GRPOLoss(nn.Module):
    def forward(self, log_probs, experience):
```

核心公式：

```
ratio = exp(log_probs - old_log_probs)        # 重要性采样比率

surr1 = ratio * advantages                     # 原始策略梯度
surr2 = clip(ratio, 0.8, 1.2) * advantages    # 裁剪后的策略梯度

loss = -mean(min(surr1, surr2)) + kl_weight * KL
```

**裁剪机制**：当 ratio 超出 `[1-ε, 1+ε]` 时，取 min 会阻止进一步更新。这和 PPO 一致。

**KL 惩罚**：`kl_weight * KL` 作为正则项，防止当前策略离参考策略太远。

**masked_mean**：只对 `action_mask=True` 的位置（即 completion 部分）计算均值，忽略 prompt 部分和 padding。

---

## 五、辅助模块

### 5.1 ReplayBuffer（replay_buffer.py）

```
Experience（一条完整序列）
    │
    ▼ split_experience_batch()
    │
12 条独立 Experience（batch_size=1）
    │
    ▼ ReplayBuffer.append()
    │
buffer = [exp1, exp2, ..., exp384]   ← 平坦列表
    │
    ▼ DataLoader + join_experience_batch()
    │
batch: Experience（batch_size=16，left-padding）
```

关键操作：
- **split_experience_batch**：用 `torch.unbind` 沿 batch 维度拆开
- **join_experience_batch**：用 `zero_pad_sequences` 做 left-padding 后 stack（生成和训练都用 left-padding）
- **ReplayBuffer**：就是一个列表，支持 `__getitem__`、`__len__`、`append`、`clear`

### 5.2 Experience 数据结构

```python
@dataclass
class Experience:
    sequences: torch.Tensor       # token ids, shape (N, seq_len)
    action_log_probs: torch.Tensor  # 旧策略 log prob, shape (N, seq_len)
    log_probs_ref: torch.Tensor   # 参考策略 log prob, shape (N, seq_len)
    returns: torch.Tensor         # reward, shape (N, 1)
    advantages: torch.Tensor      # 组内归一化后的 advantage, shape (N, 1)
    attention_mask: torch.Tensor  # 全序列的 attention mask
    action_mask: torch.Tensor     # 只标记 completion 部分（排除 prompt 和 padding）
    kl: torch.Tensor              # 当前策略 vs 参考策略的 KL 散度
```

### 5.3 序列概率计算

```python
def sequences_log_probs(model, sequence_ids, attention_mask):
    # 1. 手动构造 position_ids（从 attention_mask 累积）
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)

    # 2. 前向传播
    output = model(input_ids=sequence_ids, attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=False)

    # 3. 从 logits 提取每个 token 的 log prob
    logits = output["logits"]  # shape: (N, seq_len, vocab_size)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    log_probs = log_probs.gather(dim=-1, index=sequence_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    # shape: (N, seq_len-1)
```

---

## 六、整体流程图

```
main()
 │
 ├─ load_model()  → 4-bit Qwen + LoRA（单模型，双角色）
 ├─ 读取 math_tasks.jsonl
 │
 └─ for each step:
      │
      ├─ [Rollout, no_grad]
      │   for each prompt (x32):
      │     ├─ rollout() → generate 12 completions（LoRA 启用）
      │     ├─ 正则解析 <answer> 标签 → reward
      │     ├─ group_advantages() → 组内归一化（不需要 critic）
      │     ├─ sequences_log_probs() → 当前 log_prob（LoRA 启用）
      │     ├─ disable_adapter() → 参考 log_prob（LoRA 禁用）
      │     ├─ approx_kl_divergence() → KL
      │     └─ 存入 ReplayBuffer（共 32×12=384 条）
      │
      ├─ [Training]
      │   for each batch (16) from ReplayBuffer:
      │     ├─ sequences_log_probs() → 当前 log_prob
      │     ├─ GRPOLoss(log_probs, experience)
      │     │     ratio = exp(log_probs - old_log_probs)
      │     │     surr1 = ratio * advantages
      │     │     surr2 = clip(ratio, 0.8, 1.2) * advantages
      │     │     loss = -mean(min(surr1, surr2)) + 0.01 * KL
      │     ├─ loss.backward()
      │     ├─ clip_grad_norm_(max_norm=1.0)
      │     └─ optimizer.step()
      │
      └─ 每 20 步 save_pretrained()
```

## 七、关键设计点

- **单模型双角色**：通过 `disable_adapter()` 切换训练策略/参考策略，比加载两个模型省一半显存
- **组内归一化**：GRPO 不需要 critic 网络，直接在 group 内比较相对好坏
- **裁剪 + KL 双重约束**：PPO clip 限制单步更新幅度，KL 惩罚防止长期偏离
- **SDPA + 4-bit + gradient checkpointing**：三层显存优化叠加，让 0.5B 模型能在 8GB 显卡上训练
- **left-padding**：生成和训练都使用 left-padding，保证 position id 和 attention mask 正确对齐
