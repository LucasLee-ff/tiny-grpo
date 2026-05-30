# 测试集与 Baseline 评测

## 测试集说明

- **文件**: `data/math_test.jsonl`
- **题目数**: 41 道
- **格式**: 与训练数据一致，`id`, `question`, `answer`, `num_terms`, `num_digits`
- **难度分布**:

| 维度 | 范围 | 题数 |
|---|---|---|
| 运算类型 | 加法 | 14 |
| | 减法 | 15 |
| | 乘法 | 12 |
| | 三项运算 | 9 |
| 位数 | 1 位 | 3 |
| | 2 位 | 27 |
| | 3 位 | 11 |

- **答案验证**: 已通过 `verify_test_answers.py` 逐题用 Python `eval()` 验算，全部 41 道题答案正确。

## 评测脚本

**文件**: `eval.py`

```
用法:
    python eval.py                           # 评测基座模型
    python eval.py --checkpoint output/step_20  # 评测训练后的 checkpoint
    python eval.py --num_samples 8           # 每题采样 8 次 (默认 4)
    python eval.py -v                        # 打印每题详细结果
```

**评测指标**:

| 指标 | 含义 |
|---|---|
| exact@1 | `<answer>` 标签内答案与标准答案完全一致 |
| pass@1 | 标准答案出现在 `<answer>` 标签内（包含关系） |
| 无答案率 | 未生成 `<answer>` 标签的样本比例 |

**评测流程**:
1. 加载 4-bit Qwen2.5-0.5B-Instruct（如有 checkpoint 则加载 LoRA adapter 并 merge）
2. 对每道题用固定的 system prompt 生成 `num_samples` 次
3. 用正则 `r"<answer>(.*?)</answer>"` 提取答案
4. 统计 exact match / partial match / no answer

## Baseline 结果

**模型**: `Qwen/Qwen2.5-0.5B-Instruct` (4-bit, 未经训练)
**配置**: temperature=1.0, top_p=1.0, max_length=512, num_samples=8

```
总题目数: 41
每题采样: 8 次
总推理次数: 328

精确匹配率 (exact@1):    12/328 = 3.7%
包含答案率 (pass@1):     16/328 = 4.9%
未生成答案标签率:         298/328 = 90.9%
```

### 分析

1. **90.9% 的样本没有输出 `<answer>` 标签** — 基座 Qwen 模型没用过 DeepSeek 的 think/answer 格式，大部分回答只是纯文本叙述，如 "The sum of 321 and 654 is 975."

2. **即使模型算对了也拿不到分** — 很多样本输出了正确数值（如 `700 - 263 =` 几乎所有样本都答了 437），但因为没包在 `<answer>` 标签里，reward 为 0。这正是 GRPO 训练要解决的问题。

3. **3.7% 的 exact match 来自偶发行为** — 少数样本碰巧输出了 `<answer>` 标签且答案正确。

4. **简单题表现更好** — 1 位数的乘法（7×8、9×7）和熟悉的乘法（18×5、11×11）exact match 率更高，说明模型对常见运算有记忆。

通过 GRPO 训练后，预期 `<answer>` 标签率会大幅提升，同时数学正确率也会随奖励信号逐步提高。
