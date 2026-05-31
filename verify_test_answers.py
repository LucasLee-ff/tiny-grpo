"""验证测试集中每道题的答案是否正确，打印错误项。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import json
import re
from pathlib import Path


def compute_answer(question: str) -> str:
    """将题目文本转为算术表达式求值，返回字符串形式的整数结果。"""
    # 清理：去掉问号、等号、多余空格
    expr = question.strip()
    expr = expr.replace("?", "").replace("=", "")
    expr = expr.replace("×", "*").replace("÷", "/").replace("x", "*")
    expr = re.sub(r"\s+", " ", expr).strip()

    try:
        result = eval(expr)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result)
    except Exception as e:
        return f"EVAL_ERROR: {e}"


def main():
    test_file = Path("data/math_test.jsonl")
    problems = []
    with test_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"验证 {len(problems)} 道题...\n")
    errors = []

    for prob in problems:
        computed = compute_answer(prob["question"])
        expected = str(prob["answer"]).strip()
        if computed == expected:
            print(f"  [OK]  {prob['id']}: {prob['question']} = {expected}")
        else:
            print(f"  [ERR] {prob['id']}: {prob['question']}")
            print(f"        期望: {expected}, 计算: {computed}")
            errors.append((prob, computed))

    print(f"\n{'='*50}")
    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for prob, computed in errors:
            print(f"  {prob['id']}: {prob['question']}")
            print(f"    JSON 中的 answer: {prob['answer']}")
            print(f"    正确结果: {computed}")

        # 自动修复
        print(f"\n自动修复中...")
        for prob, computed in errors:
            prob["answer"] = computed

        with test_file.open("w", encoding="utf-8") as f:
            for prob in problems:
                f.write(json.dumps(prob, ensure_ascii=False) + "\n")
        print(f"已修复 {len(errors)} 处错误，文件已更新。")
    else:
        print("所有答案正确！")


if __name__ == "__main__":
    main()
