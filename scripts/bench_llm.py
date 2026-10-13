"""并发压测：量化「本地推理 vs 生产推理」的吞吐与延迟差异。

为什么需要这个脚本
------------------
面试里最容易被问穿的一句话是：「你说 Ollama 够用，那生产上并发怎么办？」
空口回答只能堆 PagedAttention、连续批处理这些名词。
本脚本对同一个 OpenAI 兼容端点（Ollama / vLLM / 云 API 均可）做
并发 1 / 4 / 8 的阶梯压测，输出吞吐（req/s）、P50/P95 延迟，
把结论落到**自己测出来的数字**上。

用法：
    # 1) Ollama（本地离线，单机）
    set LLM_PROVIDER=ollama && python scripts/bench_llm.py
    # 2) vLLM（生产并发）
    set LLM_PROVIDER=vllm && python scripts/bench_llm.py --concurrency 1,4,8,16 --out eval/bench_vllm.md
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.base import LLMError  # noqa: E402
from app.llm.factory import build_llm  # noqa: E402

PROMPT = "请用两句话说明：审计函证控制在什么情况下属于必须执行程序，并给出准则依据。"

TEMPLATE = """# 推理服务并发压测报告

> 本文件由 `scripts/bench_llm.py` 自动生成，请勿手工修改。

- 压测时间：{ts}
- 推理后端：`{provider}` ｜ 模型：`{model}`
- 端点：{base_url}
- 温度：{temperature} ｜ max_tokens：{max_tokens}
- 预热：{warmup} 次（不计入统计）

## 结果

| 并发数 | 成功/总请求 | 吞吐(req/s) | P50 延迟(ms) | P95 延迟(ms) | 最大延迟(ms) | 失败数 |
|---|---|---|---|---|---|---|
{rows}

## 结论口径（写简历/面试时照抄，不要超出数据）

{conclusion}
"""


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


def run_once(llm: Any) -> tuple[float, str]:
    started = time.perf_counter()
    try:
        llm.chat([{"role": "user", "content": PROMPT}])
        return ((time.perf_counter() - started) * 1000), ""
    except LLMError as exc:
        return ((time.perf_counter() - started) * 1000), str(exc)
    except Exception as exc:  # noqa: BLE001
        return ((time.perf_counter() - started) * 1000), f"{type(exc).__name__}: {exc}"


def bench(llm: Any, concurrency: int, rounds: int) -> dict[str, Any]:
    latencies: list[float] = []
    errors: list[str] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for latency, error in pool.map(lambda _: run_once(llm), range(rounds)):
            latencies.append(latency)
            if error:
                errors.append(error)
    wall = time.perf_counter() - started
    return {
        "concurrency": concurrency,
        "success": rounds - len(errors),
        "total": rounds,
        "throughput": round(rounds / wall, 2) if wall > 0 else 0.0,
        "p50": int(percentile(latencies, 0.50)),
        "p95": int(percentile(latencies, 0.95)),
        "max": int(max(latencies)) if latencies else 0,
        "errors": len(errors),
        "first_error": errors[0] if errors else "",
    }


def build_conclusion(results: list[dict[str, Any]], provider: str) -> str:
    if not results:
        return "- 无有效数据。"
    base = results[0]
    peak = max(results, key=lambda item: item["throughput"])
    lines = [
        f"1. 并发从 {base['concurrency']} 提升到 {peak['concurrency']}，吞吐 {base['throughput']} → {peak['throughput']} req/s，"
        f"P95 延迟 {base['p95']} → {peak['p95']} ms。",
        "2. 延迟随并发上升属预期：推理服务会排队。吞吐与延迟必须成对看，只报吞吐是不诚实的。",
    ]
    if provider == "ollama":
        lines.append(
            "3. Ollama 的定位是**本地离线单机**：零依赖、模型管理简单，适合涉密数据不出机与开发调试；"
            "但它逐请求生成、缺少连续批处理与 PagedAttention 式显存管理，并发上不去。"
        )
    elif provider == "vllm":
        lines.append(
            "3. vLLM 通过 PagedAttention + 连续批处理在并发下显著优于逐请求生成，"
            "生产应把 Ollama 换成 vLLM（同样使用本地权重，数据仍不出机）。"
        )
    lines.append("4. 面试表述：本地化解决的是**合规**问题，vLLM 解决的是**并发**问题，两者不冲突，可叠加部署。")
    return "\n".join(f"- {line}" for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="推理服务并发压测")
    parser.add_argument("--concurrency", default="1,4,8", help="并发阶梯，逗号分隔")
    parser.add_argument("--rounds", type=int, default=8, help="每个并发档位的请求数")
    parser.add_argument("--warmup", type=int, default=1, help="预热请求数")
    parser.add_argument("--tag", default="", help="写入报告的标签，如 ollama-9b")
    parser.add_argument("--out", default="", help="报告输出路径；为空则打印到控制台")
    args = parser.parse_args()

    settings = get_settings()
    llm = build_llm(settings)
    print(f"[bench] provider={settings.provider} model={settings.resolved_chat_model} base_url={settings.resolved_base_url or '—'}")
    if settings.provider == "mock":
        print("[bench] ⚠ 当前为 mock 后端，测的是本地调用开销，不代表真实推理性能；结论仅用于验证脚本链路。")
    print(f"[bench] health={llm.health()}")

    for _ in range(max(0, args.warmup)):
        run_once(llm)

    levels = [int(item) for item in args.concurrency.split(",") if item.strip().isdigit()]
    results: list[dict[str, Any]] = []
    for level in levels:
        result = bench(llm, level, args.rounds)
        results.append(result)
        print(
            f"[bench] 并发 {result['concurrency']:>2} | 吞吐 {result['throughput']:>6} req/s | "
            f"P50 {result['p50']:>5}ms | P95 {result['p95']:>5}ms | 失败 {result['errors']}"
        )
        if result["first_error"]:
            print(f"        首个错误：{result['first_error'][:120]}")

    rows = "\n".join(
        f"| {r['concurrency']} | {r['success']}/{r['total']} | {r['throughput']} | {r['p50']} | {r['p95']} | {r['max']} | {r['errors']} |"
        for r in results
    )
    report = TEMPLATE.format(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        provider=settings.provider,
        model=f"{settings.resolved_chat_model}{(' · ' + args.tag) if args.tag else ''}",
        base_url=settings.resolved_base_url or "本地进程内",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        warmup=args.warmup,
        rows=rows,
        conclusion=build_conclusion(results, settings.provider),
    )
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"[bench] 报告已写入：{target}")
    else:
        print()
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
