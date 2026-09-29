"""
Tạo báo cáo kết quả đánh giá RAG Pipeline.
Hỗ trợ 3 định dạng: Console, JSON, Markdown.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from evaluation.config import THRESHOLDS, LOWER_IS_BETTER, RESULTS_DIR, METRIC_GROUPS


def _pass_or_fail(metric_name: str, value: float) -> str:
    """Kiểm tra metric có đạt ngưỡng không."""
    threshold = THRESHOLDS.get(metric_name)
    if threshold is None:
        return "N/A"
    if metric_name in LOWER_IS_BETTER:
        return "✅ PASS" if value <= threshold else "❌ FAIL"
    else:
        return "✅ PASS" if value >= threshold else "❌ FAIL"


def _format_value(metric_name: str, value: float) -> str:
    """Format giá trị metric cho dễ đọc."""
    if metric_name == "average_latency":
        return f"{value:.2f}s"
    elif metric_name == "hallucination_rate":
        return f"{value:.1%}"
    else:
        return f"{value:.4f}"


def print_console_report(
    aggregated_results: Dict[str, float],
    per_question_results: Optional[List[Dict]] = None,
    category_results: Optional[Dict[str, Dict]] = None
):
    """
    In báo cáo kết quả ra console với bảng và màu sắc.
    """
    print("\n" + "=" * 70)
    print(" BÁO CÁO ĐÁNH GIÁ RAG PIPELINE ".center(70, "="))
    print(f" Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(70, " "))
    print("=" * 70)

    # Đếm pass/fail
    total_metrics = 0
    passed = 0
    failed = 0

    for group_name, group_label in [
        ("retrieval", "📦 RETRIEVAL METRICS"),
        ("generation", "🤖 GENERATION METRICS"),
        ("system", "🏗️  SYSTEM METRICS")
    ]:
        metrics = METRIC_GROUPS.get(group_name, [])
        print(f"\n{'─' * 70}")
        print(f"  {group_label}")
        print(f"{'─' * 70}")
        print(f"  {'Metric':<25} {'Value':>10} {'Threshold':>12} {'Status':>10}")
        print(f"  {'─' * 57}")

        for metric in metrics:
            value = aggregated_results.get(metric)
            if value is None:
                print(f"  {metric:<25} {'N/A':>10} {'':>12} {'⚠️ SKIP':>10}")
                continue

            threshold = THRESHOLDS.get(metric)
            status = _pass_or_fail(metric, value)
            formatted_value = _format_value(metric, value)

            if threshold is not None:
                if metric in LOWER_IS_BETTER:
                    threshold_str = f"≤ {_format_value(metric, threshold)}"
                else:
                    threshold_str = f"≥ {_format_value(metric, threshold)}"
            else:
                threshold_str = "N/A"

            print(f"  {metric:<25} {formatted_value:>10} {threshold_str:>12} {status:>10}")
            total_metrics += 1
            if "PASS" in status:
                passed += 1
            elif "FAIL" in status:
                failed += 1

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  TỔNG KẾT: {passed}/{total_metrics} metrics PASS, {failed}/{total_metrics} metrics FAIL")
    pass_rate = (passed / total_metrics * 100) if total_metrics > 0 else 0
    print(f"  Tỷ lệ đạt: {pass_rate:.1f}%")
    print(f"{'=' * 70}")

    # In phân tích theo category nếu có
    if category_results:
        print(f"\n{'─' * 70}")
        print("  📊 PHÂN TÍCH THEO LOẠI CÂU HỎI")
        print(f"{'─' * 70}")
        print(f"  {'Category':<15} {'Count':>6} {'Avg Recall':>12} {'Avg Faith.':>12} {'Avg Latency':>12}")
        print(f"  {'─' * 57}")
        for cat, cat_data in category_results.items():
            count = cat_data.get("count", 0)
            avg_recall = cat_data.get("avg_recall_at_5", 0)
            avg_faith = cat_data.get("avg_faithfulness", 0)
            avg_lat = cat_data.get("avg_latency", 0)
            print(f"  {cat:<15} {count:>6} {avg_recall:>12.4f} {avg_faith:>12.4f} {avg_lat:>11.2f}s")


def save_json_report(
    aggregated_results: Dict[str, float],
    per_question_results: List[Dict],
    category_results: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Lưu kết quả đánh giá dưới dạng JSON.
    Returns: đường dẫn file đã lưu.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(RESULTS_DIR / f"eval_results_{timestamp}.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "aggregated_metrics": aggregated_results,
        "thresholds": THRESHOLDS,
        "pass_fail": {
            metric: _pass_or_fail(metric, value)
            for metric, value in aggregated_results.items()
        },
        "per_question_results": per_question_results,
    }

    if category_results:
        report["category_analysis"] = category_results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Đã lưu báo cáo JSON: {output_path}")
    return output_path


def save_markdown_report(
    aggregated_results: Dict[str, float],
    per_question_results: List[Dict],
    category_results: Optional[Dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Lưu kết quả đánh giá dưới dạng Markdown.
    Returns: đường dẫn file đã lưu.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(RESULTS_DIR / f"eval_report_{timestamp}.md")

    lines = []
    lines.append("# 📊 Báo Cáo Đánh Giá RAG Pipeline - Chatbot Luật Việt Nam\n")
    lines.append(f"**Thời gian đánh giá:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Số câu hỏi test:** {len(per_question_results)}\n")

    # Tổng hợp metrics
    lines.append("## 1. Tổng Hợp Chỉ Số\n")

    for group_name, group_label in [
        ("retrieval", "### 📦 Retrieval Metrics"),
        ("generation", "### 🤖 Generation Metrics"),
        ("system", "### 🏗️ System Metrics")
    ]:
        metrics = METRIC_GROUPS.get(group_name, [])
        lines.append(f"\n{group_label}\n")
        lines.append("| Metric | Giá trị | Ngưỡng | Kết quả |")
        lines.append("|--------|---------|--------|---------|")

        for metric in metrics:
            value = aggregated_results.get(metric)
            if value is None:
                lines.append(f"| {metric} | N/A | N/A | ⚠️ SKIP |")
                continue

            threshold = THRESHOLDS.get(metric)
            status = _pass_or_fail(metric, value)
            formatted_value = _format_value(metric, value)

            if threshold is not None:
                if metric in LOWER_IS_BETTER:
                    threshold_str = f"≤ {_format_value(metric, threshold)}"
                else:
                    threshold_str = f"≥ {_format_value(metric, threshold)}"
            else:
                threshold_str = "N/A"

            lines.append(f"| {metric} | {formatted_value} | {threshold_str} | {status} |")

    # Category analysis
    if category_results:
        lines.append("\n## 2. Phân Tích Theo Loại Câu Hỏi\n")
        lines.append("| Category | Số lượng | Avg Recall@5 | Avg Faithfulness | Avg Latency |")
        lines.append("|----------|----------|--------------|------------------|-------------|")
        for cat, cat_data in category_results.items():
            count = cat_data.get("count", 0)
            avg_recall = cat_data.get("avg_recall_at_5", 0)
            avg_faith = cat_data.get("avg_faithfulness", 0)
            avg_lat = cat_data.get("avg_latency", 0)
            lines.append(f"| {cat} | {count} | {avg_recall:.4f} | {avg_faith:.4f} | {avg_lat:.2f}s |")

    # Per-question details
    lines.append("\n## 3. Chi Tiết Từng Câu Hỏi\n")
    for result in per_question_results:
        test_id = result.get("id", "?")
        question = result.get("question", "")
        category = result.get("category", "")
        lines.append(f"\n### {test_id} ({category})")
        lines.append(f"**Câu hỏi:** {question}\n")

        retrieval = result.get("retrieval_metrics", {})
        generation = result.get("generation_metrics", {})
        system = result.get("system_metrics", {})

        lines.append("| Nhóm | Metric | Giá trị |")
        lines.append("|------|--------|---------|")

        for metric, value in retrieval.items():
            lines.append(f"| Retrieval | {metric} | {value:.4f} |")
        for metric, value in generation.items():
            lines.append(f"| Generation | {metric} | {value:.4f} |")
        for metric, value in system.items():
            if isinstance(value, float):
                lines.append(f"| System | {metric} | {value:.4f} |")
            else:
                lines.append(f"| System | {metric} | {value} |")

    # Summary
    passed = sum(
        1 for m, v in aggregated_results.items()
        if "PASS" in _pass_or_fail(m, v)
    )
    total = len(aggregated_results)
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    lines.append(f"\n---\n")
    lines.append(f"## Tổng kết\n")
    lines.append(f"- **PASS:** {passed}/{total} metrics")
    lines.append(f"- **FAIL:** {failed}/{total} metrics")
    lines.append(f"- **Tỷ lệ đạt:** {pass_rate:.1f}%")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"📝 Đã lưu báo cáo Markdown: {output_path}")
    return output_path
