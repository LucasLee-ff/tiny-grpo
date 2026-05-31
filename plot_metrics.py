import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""读取 metrics.csv 并生成趋势图。

用法:
    python plot_metrics.py                  # 画当前目录的 metrics.csv
    python plot_metrics.py -f run1/metrics.csv  # 指定文件
    python plot_metrics.py --smooth 5       # 用窗口=5 做移动平均平滑
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def moving_average(data: pd.Series, window: int) -> pd.Series:
    return data.rolling(window=window, min_periods=1).mean()


def main():
    parser = argparse.ArgumentParser(description="绘制训练指标趋势图")
    parser.add_argument("-f", "--file", default="metrics.csv", help="CSV 文件路径")
    parser.add_argument("--smooth", type=int, default=0, help="移动平均窗口大小 (0=不平滑)")
    parser.add_argument("-o", "--output", default="metrics.png", help="输出图片路径")
    args = parser.parse_args()

    csv_path = Path(args.file)
    if not csv_path.exists():
        print(f"文件不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Metrics ({csv_path.name})", fontsize=14)

    metrics = [
        ("returns", "Returns (total per step)", "green"),
        ("loss", "Loss", "red"),
        ("kl", "KL Divergence", "blue"),
        ("grad_norm", "Gradient Norm", "purple"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"no '{col}' column", ha="center", va="center")
            continue

        series = df[col].dropna()
        ax.plot(series.index, series.values, color=color, alpha=0.3, linewidth=0.8)

        if args.smooth > 0 and len(series) > args.smooth:
            smoothed = moving_average(series, args.smooth)
            ax.plot(smoothed.index, smoothed.values, color=color, linewidth=1.5,
                    label=f"smooth={args.smooth}")

        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        if args.smooth > 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"图片已保存: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
