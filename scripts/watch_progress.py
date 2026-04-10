"""
实时进度监控工具 — Phase 5.2 全量生成
用法 (从项目根目录):
    python scripts/watch_progress.py          # 每 10 秒刷新
    python scripts/watch_progress.py --once   # 只打印一次
"""
import os
import sys
import time
import json
import re
import argparse
from datetime import datetime, timedelta

# 始终相对于项目根目录解析路径
ROOT = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.abspath(ROOT)

RAG_PATH    = os.path.join(ROOT, "results", "explanations_rag.jsonl")
PROMPT_PATH = os.path.join(ROOT, "results", "explanations_prompt_only.jsonl")
LOG_PATH    = os.path.join(ROOT, "logs", "phase52_full.log")
TOTAL_PAIRS = 59500   # 5950 users × 10
TOTAL_LLM   = TOTAL_PAIRS * 2  # RAG + prompt-only


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def tail_log(path, n=5):
    if not os.path.exists(path):
        return ["(日志文件尚未创建)"]
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:] if l.strip()]


def parse_log_speed(path):
    """从日志里抓最新的 tqdm 速率和 ETA。"""
    if not os.path.exists(path):
        return None, None
    speed = None
    eta   = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        matches = re.findall(
            r'(\d+)/59500 \[[\d:]+<([\d:]+),\s*([\d.]+)it/s\]', content)
        if matches:
            last = matches[-1]
            speed = float(last[2])
            eta_str = last[1]
            parts = eta_str.split(":")
            if len(parts) == 2:
                eta = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                eta = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return speed, eta


def bar(done, total, width=38):
    frac   = done / total if total > 0 else 0
    filled = int(frac * width)
    pct    = frac * 100
    return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1f}%"


def render(once=False):
    while True:
        rag_done    = count_lines(RAG_PATH)
        prompt_done = count_lines(PROMPT_PATH)
        total_done  = rag_done + prompt_done

        speed, eta = parse_log_speed(LOG_PATH)

        if eta is not None:
            eta_str = str(timedelta(seconds=eta))
        elif speed and speed > 0:
            remaining = TOTAL_PAIRS - rag_done
            eta_str = str(timedelta(seconds=int(remaining / speed)))
        else:
            eta_str = "计算中…"

        now = datetime.now().strftime("%H:%M:%S")

        if not once:
            os.system("clear")

        print("=" * 62)
        print(f"  Phase 5.2 全量生成进度         更新: {now}")
        print("=" * 62)
        print(f"\n  总进度  {total_done:,} / {TOTAL_LLM:,} 次 LLM 调用")
        print(f"  {bar(total_done, TOTAL_LLM, 50)}")
        print()
        print(f"  RAG 解释      {bar(rag_done, TOTAL_PAIRS)}  {rag_done:>6,}/{TOTAL_PAIRS:,}")
        print(f"  Prompt-only   {bar(prompt_done, TOTAL_PAIRS)}  {prompt_done:>6,}/{TOTAL_PAIRS:,}")
        print()
        if speed:
            print(f"  速度:    {speed:.2f} pairs/s  ({speed * 60:.0f} pairs/min)")
        print(f"  预计剩余: {eta_str}")
        print()
        print(f"  日志: logs/phase52_full.log")
        print("-" * 62)
        print("  最新日志:")
        for line in tail_log(LOG_PATH, 4):
            if len(line) > 58:
                line = line[:55] + "..."
            print(f"    {line}")
        print("=" * 62)

        if once:
            break

        print("\n  (每 10 秒刷新，Ctrl+C 退出)\n")
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n已退出监控（后台任务仍在继续运行）")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="只打印一次")
    args = parser.parse_args()
    render(once=args.once)
