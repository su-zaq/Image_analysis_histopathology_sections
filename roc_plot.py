"""評価用の ROC 曲線計算・保存ユーティリティ。"""
from __future__ import annotations

import os

import cv2
import numpy as np


def compute_pixel_roc(
    pred_img: np.ndarray,
    ans_img: np.ndarray,
    ans_threshold: int = 127,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """推論画像の連続値と正解の二値ラベルからピクセル単位の ROC を計算する。

    Args:
        pred_img: グレースケール推論画像 (0-255)
        ans_img: グレースケール正解画像
        ans_threshold: 正解を二値化する閾値

    Returns:
        (fpr, tpr, auc)。正例または負例が無い場合は (None, None, None)。
    """
    if pred_img is None or ans_img is None:
        return None, None, None

    if pred_img.shape != ans_img.shape:
        ans_img = cv2.resize(ans_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    scores = pred_img.ravel().astype(np.float64) / 255.0
    labels = (ans_img.ravel() >= ans_threshold).astype(np.uint8)

    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None, None, None

    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]

    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)

    tpr = np.concatenate(([0.0], tp / positive_count))
    fpr = np.concatenate(([0.0], fp / negative_count))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def save_roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: str,
    title: str | None = None,
) -> None:
    """ROC 曲線を PNG として保存する。"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color='#1f77b4', linewidth=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_roc_from_images(
    pred_img: np.ndarray,
    ans_img: np.ndarray,
    save_path: str,
    title: str | None = None,
    ans_threshold: int = 127,
) -> bool:
    """画像ペアから ROC 曲線を計算して保存する。"""
    fpr, tpr, auc = compute_pixel_roc(pred_img, ans_img, ans_threshold=ans_threshold)
    if fpr is None or tpr is None or auc is None:
        return False
    save_roc_curve_plot(fpr, tpr, auc, save_path, title=title)
    return True


def plot_roc_from_image_pairs(
    image_pairs: list[tuple[np.ndarray, np.ndarray]],
    save_path: str,
    title: str | None = None,
    ans_threshold: int = 127,
) -> bool:
    """複数画像のピクセルを結合した ROC 曲線を保存する。"""
    score_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for pred_img, ans_img in image_pairs:
        if pred_img is None or ans_img is None:
            continue
        if pred_img.shape != ans_img.shape:
            ans_img = cv2.resize(
                ans_img,
                (pred_img.shape[1], pred_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        score_chunks.append(pred_img.ravel().astype(np.float64) / 255.0)
        label_chunks.append((ans_img.ravel() >= ans_threshold).astype(np.uint8))

    if not score_chunks:
        return False

    scores = np.concatenate(score_chunks)
    labels = np.concatenate(label_chunks)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return False

    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = np.concatenate(([0.0], tp / positive_count))
    fpr = np.concatenate(([0.0], fp / negative_count))
    auc = float(np.trapz(tpr, fpr))
    save_roc_curve_plot(fpr, tpr, auc, save_path, title=title)
    return True
