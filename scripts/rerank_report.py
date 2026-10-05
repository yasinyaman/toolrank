"""The tables of docs/reports/faz2-rerank.md, from the result files: what a second stage (CLM,
cross-encoders) and a LoRA-trained backbone do to the packaged heads' rankings on ToolRet,
LiveMCPBench and MCP-Zero. Base rows come from docs/results/ (the README's), the rest from
results/ (fetched from the GB10: clm_*, clmj_*, cross_*, crosslora_*, lora_*). A hosted second
stage was measured too; its provider's terms keep its results out of the published report.

    uv run python scripts/rerank_report.py            # print the summary and the tables
    uv run python scripts/rerank_report.py --write    # splice them into the report, between the markers

Every header carries the direction and the unit (↑ higher is better, ↓ lower; % = percent), as the
report's legend says. Rows whose file is missing are left out, so the script runs while rows are
still arriving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "reports" / "faz2-rerank.md"
SETS = [("toolret", "ToolRet"), ("livemcpbench_server", "LiveMCPBench"), ("mcp_zero_server", "MCP-Zero")]
ROWS = [
    ("BM25 (w/ inst)", "readme_{s}_bm25_inst"),
    ("Qwen3-Embedding-8B", "readme_{s}_qwen3emb"),
    ("heads", "readme_{s}_heads"),
    ("heads → CLM-8B, ilk 100", "clm_{s}_heads_clm100"),
    ("heads → CLM-8B, ilk 20", "clm_{s}_heads_clm20"),
    ("heads → CLM fine-tune, ilk 100", "clm_{s}_heads_clmft100"),
    ("BM25 → CLM-8B, ilk 30", "clm_{s}_bm25_clm30"),
    ("heads → CLM-8B, kesilmiş metin, ilk 100", "clmj_{s}_heads_clm100"),
    ("heads → CLM-8B, kesilmiş metin, ilk 20, documentation", "clmj_{s}_heads_clm20doc"),
    ("BM25 → CLM-8B, kesilmiş metin, ilk 30", "clmj_{s}_bm25_clm30"),
    ("heads → Qwen3-Reranker-8B, ilk 20, documentation", "cross_{s}_qwen3_heads_x20doc"),
    ("heads → Qwen3-Reranker-8B, ilk 100", "cross_{s}_qwen3_heads_x100"),
    ("BM25 → Qwen3-Reranker-8B, ilk 30", "cross_{s}_qwen3_bm25_x30"),
    ("Qwen3-Reranker-8B tek başına", "cross_{s}_qwen3_alone"),
    ("heads → bge-reranker-v2-gemma, ilk 20, documentation", "cross_{s}_bge_heads_x20doc"),
    ("heads → bge-reranker-v2-gemma, ilk 100", "cross_{s}_bge_heads_x100"),
    ("BM25 → bge-reranker-v2-gemma, ilk 30", "cross_{s}_bge_bm25_x30"),
    ("bge-reranker-v2-gemma tek başına", "cross_{s}_bge_alone"),
    ("Qwen3-Emb + LoRA", "lora_{s}_lora"),
    ("Qwen3-Emb + LoRA + head (epoch 0 = kimlik)", "lora_{s}_lora_heads"),
    ("LoRA → Qwen3-Reranker-8B, ilk 20, documentation", "crosslora_{s}_qwen3_heads_x20doc"),
    ("LoRA → Qwen3-Reranker-8B, ilk 100", "crosslora_{s}_qwen3_heads_x100"),
]
SUMMARY = [
    ("heads (Qwen3-Embedding-8B + v0.1 head'leri), tek aşama", "readme_{s}_heads", "~1 ms, yerel"),
    ("Qwen3-Embedding-8B + LoRA, tek aşama", "lora_{s}_lora", "~1 ms, yerel; MCP-Zero seçim seti"),
    (
        "heads → Qwen3-Reranker-8B, ilk 20 + dokümantasyon",
        "cross_{s}_qwen3_heads_x20doc",
        "+1.7–2.5 s (yük altında), yerel, +16 GB",
    ),
    (
        "LoRA → Qwen3-Reranker-8B, ilk 20 + dokümantasyon",
        "crosslora_{s}_qwen3_heads_x20doc",
        "+1.7–2.5 s (yük altında), yerel, +16 GB",
    ),
    (
        "heads → bge-reranker-v2-gemma, ilk 20 + dokümantasyon",
        "cross_{s}_bge_heads_x20doc",
        "+0.1 s, yerel, +5 GB",
    ),
    ("heads → CLM-8B, ilk 20", "clm_{s}_heads_clm20", "+2 ms, yerel"),
]


def load(name: str) -> dict | None:
    for d in ("results", "docs/results"):
        p = ROOT / d / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def set_table(s: str, title: str) -> list[str]:
    base = load(f"readme_{s}_heads")
    ms = ["NDCG@10", "Recall@5", "Recall@10", "Comprehensiveness@10"] + (
        ["Precision@1"] if s == "mcp_zero_server" else []
    )
    cat = s == "toolret"
    heads = [f"{m} ↑ %" for m in ms] + (["cat-macro ↑ %"] if cat else [])
    out = [f"### {title} (w/ inst, n={base['n_queries']})", ""]
    out.append("| Satır | " + " | ".join(heads) + " | sorgu p50 ↓ ms | çağrı p50 ↓ ms | çift / sorgu ↓ |")
    out.append("|---|" + "---:|" * (len(heads) + 3))
    for label, key in ROWS:
        d = load(key.format(s=s))
        if d is None:
            continue
        cells = [pct(d["overall"][m]) for m in ms]
        if cat:
            cells.append(pct(d["category_macro"]["NDCG@10"]))
        cells.append(f"{d['latency_ms']['per_query_p50']:.1f}")
        x, n = d["config"].get("cross"), d["n_queries"]
        if x:
            p50 = f"{x['call_ms_p50']:.0f}" if x.get("call_ms_p50") else "önbellek"
            cells += [p50, f"{(x['pairs'] + x['cached_pairs']) / n:,.0f} çift"]
        else:
            cells += ["—", "—"]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out + [""]


def moves(a_key: str, b_key: str, a_name: str, b_name: str) -> list[str]:
    a, b = load(a_key), load(b_key)
    if a is None or b is None:
        return []
    out = [f"### ToolRet, kategori bazında NDCG@10: {a_name} → {b_name}", ""]
    out += [f"| Kategori | {a_name} ↑ % | {b_name} ↑ % | fark ↑ puan |", "|---|---:|---:|---:|"]
    for c in a["per_category"]:
        x, y = a["per_category"][c]["NDCG@10"], b["per_category"][c]["NDCG@10"]
        out.append(f"| {c} | {pct(x)} | {pct(y)} | {100 * (y - x):+.2f} |")
    mv = sorted((100 * (b["per_task"][t]["NDCG@10"] - a["per_task"][t]["NDCG@10"]), t) for t in a["per_task"])
    out += ["", f"### ToolRet, görev bazında en büyük hareketler (NDCG@10): {a_name} → {b_name}", ""]
    out += [f"| Görev | n | {a_name} ↑ % | {b_name} ↑ % | fark ↑ puan |", "|---|---:|---:|---:|---:|"]
    for d, t in mv[:6] + mv[-6:]:
        n = a["config"]["task_counts"].get(t, 0)
        out.append(
            f"| {t} | {n} | {pct(a['per_task'][t]['NDCG@10'])} | {pct(b['per_task'][t]['NDCG@10'])} | {d:+.2f} |"
        )
    up, down = sum(1 for d, _ in mv if d > 0.5), sum(1 for d, _ in mv if d < -0.5)
    return out + ["", f"{len(mv)} görevden {up} yukarı, {down} aşağı (0,5 puandan fazla).", ""]


def tables() -> str:
    out: list[str] = []
    for s, title in SETS:
        out += set_table(s, title)
    out += moves("readme_toolret_heads", "lora_toolret_lora", "heads", "LoRA")
    return "\n".join(out).rstrip() + "\n"


def summary() -> str:
    out = [
        "| Düzen | ToolRet NDCG@10 ↑ % | ToolRet cat-macro ↑ % | LiveMCPBench NDCG@10 ↑ % | MCP-Zero Precision@1 ↑ % | sorgu başına bedel |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, key, cost in SUMMARY:
        t, lv, mz = (load(key.format(s=s)) for s, _ in SETS)
        if t is None and lv is None and mz is None:
            continue
        cells = [
            pct(t["overall"]["NDCG@10"]) if t else "—",
            pct(t["category_macro"]["NDCG@10"]) if t else "—",
            pct(lv["overall"]["NDCG@10"]) if lv else "—",
            pct(mz["overall"]["Precision@1"]) if mz else "—",
        ]
        out.append(f"| {label} | " + " | ".join(cells) + f" | {cost} |")
    return "\n".join(out) + "\n"


def splice(text: str, name: str, body: str) -> str:
    a, b = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.index(a) + len(a), text.index(b)
    return text[:i] + "\n" + body + text[j:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--write", action="store_true", help="splice the summary and the tables into the report")
    a = p.parse_args()
    if not a.write:
        print(summary())
        print(tables())
        return 0
    text = REPORT.read_text()
    text = splice(splice(text, "summary", summary()), "tables", tables())
    REPORT.write_text(text)
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
