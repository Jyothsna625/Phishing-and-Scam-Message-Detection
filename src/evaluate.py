"""Model evaluation metrics and Pillow-based chart generation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, METRICS_SUMMARY_PATH


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    """Load a clean TrueType font if available, else fall back to default."""
    for font_name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Compute standard binary classification metrics and ROC curve points."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(y), 4) for y in tpr],
        },
    }


def save_model_comparison_chart(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Draw a grouped bar chart comparing candidate models across key metrics."""
    width, height = 900, 520
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(22)
    label_font = _load_font(15)
    small_font = _load_font(13)

    draw.text((40, 24), "Model Performance Comparison", fill="#0f172a", font=title_font)

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    colors = ["#2563eb", "#0ea5e9", "#10b981", "#f59e0b"]

    chart_left, chart_top = 80, 95
    chart_right, chart_bottom = 850, 430
    chart_height = chart_bottom - chart_top

    # Horizontal grid lines
    for step in range(6):
        val = step * 0.2
        y = int(chart_bottom - val * chart_height)
        draw.line([(chart_left, y), (chart_right, y)], fill="#e2e8f0", width=1)
        draw.text((34, y - 8), f"{val:.1f}", fill="#475569", font=small_font)

    draw.line([(chart_left, chart_top), (chart_left, chart_bottom)], fill="#334155", width=2)
    draw.line([(chart_left, chart_bottom), (chart_right, chart_bottom)], fill="#334155", width=2)

    model_names = list(results.keys())
    group_width = (chart_right - chart_left) / len(metrics)
    bar_width = min(46, int((group_width * 0.68) / max(len(model_names), 1)))

    for m_idx, (metric_key, metric_label) in enumerate(zip(metrics, metric_labels)):
        group_center = chart_left + group_width * (m_idx + 0.5)
        total_bars_width = bar_width * len(model_names) + 8 * (len(model_names) - 1)
        start_x = group_center - total_bars_width / 2

        for model_idx, model_name in enumerate(model_names):
            score = results[model_name][metric_key]
            bx0 = int(start_x + model_idx * (bar_width + 8))
            bx1 = bx0 + bar_width
            by1 = chart_bottom
            by0 = int(chart_bottom - score * chart_height)
            color = colors[model_idx % len(colors)]
            draw.rectangle([bx0, by0, bx1, by1], fill=color)
            draw.text((bx0 + 2, max(chart_top - 18, by0 - 18)), f"{score:.2f}", fill="#0f172a", font=small_font)

        draw.text((int(group_center - 32), chart_bottom + 14), metric_label, fill="#1e293b", font=label_font)

    # Legend
    legend_x = 80
    for idx, model_name in enumerate(model_names):
        color = colors[idx % len(colors)]
        draw.rectangle([legend_x, 475, legend_x + 18, 493], fill=color)
        draw.text((legend_x + 26, 474), model_name, fill="#0f172a", font=label_font)
        legend_x += 230

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_confusion_matrix_chart(
    cm: list[list[int]],
    model_name: str,
    output_path: Path,
) -> None:
    """Draw a 2x2 confusion matrix heatmap PNG for the best model."""
    width, height = 620, 520
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(20)
    label_font = _load_font(15)
    num_font = _load_font(28)

    draw.text((40, 24), f"Confusion Matrix - {model_name}", fill="#0f172a", font=title_font)

    grid_left, grid_top = 180, 120
    cell_size = 150
    max_val = max(max(row) for row in cm) or 1
    classes = ["Legitimate", "Phishing"]

    for r_idx in range(2):
        for c_idx in range(2):
            val = cm[r_idx][c_idx]
            intensity = val / max_val
            # Interpolate from light slate (#eff6ff) to deep blue (#1d4ed8)
            red = int(239 - intensity * (239 - 29))
            green = int(246 - intensity * (246 - 78))
            blue = int(255 - intensity * (255 - 216))
            x0 = grid_left + c_idx * cell_size
            y0 = grid_top + r_idx * cell_size
            draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=(red, green, blue), outline="#94a3b8", width=2)
            text_color = "#ffffff" if intensity > 0.55 else "#0f172a"
            draw.text((x0 + 58, y0 + 56), str(val), fill=text_color, font=num_font)

    for idx, cls_name in enumerate(classes):
        draw.text((grid_left + idx * cell_size + 34, grid_top + 2 * cell_size + 18), cls_name, fill="#1e293b", font=label_font)
        draw.text((70, grid_top + idx * cell_size + 65), cls_name, fill="#1e293b", font=label_font)

    draw.text((grid_left + 95, grid_top + 2 * cell_size + 52), "Predicted Label", fill="#0f172a", font=label_font)
    draw.text((22, grid_top + 130), "Actual", fill="#0f172a", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_roc_curve_chart(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Draw ROC curves comparing candidate models."""
    width, height = 680, 520
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(20)
    label_font = _load_font(14)
    small_font = _load_font(12)

    draw.text((40, 22), "ROC Curve Comparison", fill="#0f172a", font=title_font)

    left, top, right, bottom = 90, 80, 610, 430
    plot_w = right - left
    plot_h = bottom - top

    for step in range(6):
        ratio = step * 0.2
        gx = int(left + ratio * plot_w)
        gy = int(bottom - ratio * plot_h)
        draw.line([(left, gy), (right, gy)], fill="#e2e8f0", width=1)
        draw.line([(gx, top), (gx, bottom)], fill="#e2e8f0", width=1)
        draw.text((left - 36, gy - 8), f"{ratio:.1f}", fill="#475569", font=small_font)
        draw.text((gx - 10, bottom + 8), f"{ratio:.1f}", fill="#475569", font=small_font)

    draw.rectangle([left, top, right, bottom], outline="#334155", width=2)
    # Random guess diagonal
    draw.line([(left, bottom), (right, top)], fill="#94a3b8", width=2)

    colors = ["#2563eb", "#10b981", "#f59e0b"]
    for idx, (model_name, metrics) in enumerate(results.items()):
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        points = [(int(left + x * plot_w), int(bottom - y * plot_h)) for x, y in zip(fpr, tpr)]
        if len(points) >= 2:
            draw.line(points, fill=colors[idx % len(colors)], width=3)

        legend_y = 462 + idx * 22
        draw.line([(left, legend_y + 7), (left + 28, legend_y + 7)], fill=colors[idx % len(colors)], width=3)
        draw.text(
            (left + 38, legend_y),
            f"{model_name} (AUC = {metrics['roc_auc']:.4f})",
            fill="#0f172a",
            font=label_font,
        )

    draw.text((300, bottom + 28), "False Positive Rate", fill="#0f172a", font=label_font)
    draw.text((16, 240), "TPR", fill="#0f172a", font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def save_evaluation_reports(
    results: dict[str, dict[str, Any]],
    best_model_name: str,
) -> Path:
    """Persist metrics summary JSON and generated PNG charts."""
    REPORTS_DIR = METRICS_SUMMARY_PATH.parent
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_model": best_model_name,
        "models": results,
    }
    METRICS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_model_comparison_chart(results, FIGURES_DIR / "model_comparison.png")
    save_confusion_matrix_chart(
        results[best_model_name]["confusion_matrix"],
        best_model_name,
        FIGURES_DIR / "confusion_matrix.png",
    )
    save_roc_curve_chart(results, FIGURES_DIR / "roc_curve.png")
    return METRICS_SUMMARY_PATH
