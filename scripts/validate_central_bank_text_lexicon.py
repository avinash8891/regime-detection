"""Validate the v2 §2A central-bank-text lexicon against the
`gtfintechlab/fomc_communication` labeled corpus.

Audit follow-up to V2 §2A Ambiguity Log entry #72. This script measures
how well the deterministic hawkish/dovish lexicon (shipped in
`regime_detection.central_bank_text`) discriminates against the Shah,
Paturi & Chava 2023 "Trillion Dollar Words" labeled FOMC corpus
(EMNLP 2023; https://huggingface.co/datasets/gtfintechlab/fomc_communication).

Labels in the corpus:
    0 = dovish
    1 = hawkish
    2 = neutral

The lexicon's ``score_text()`` returns ``net_score``:
    NaN   → no lexicon hits        → mapped to neutral (2)
    > 0   → more hawkish hits      → mapped to hawkish (1)
    < 0   → more dovish hits       → mapped to dovish (0)
    = 0   → equal hawkish/dovish   → mapped to neutral (2)

Two validations are reported:

1. **Sentence-level** — score each labeled sentence directly. Strict
   test of discrimination at fine granularity; expect modest accuracy
   because most short sentences have no lexicon hits (the lexicon was
   designed for full-document scoring).

2. **Document-level (pooled by year)** — concatenate all sentences in
   each (year, label) cell, score the pooled text, compare its
   net_score sign to the cell's label. Closer to how the engine
   actually uses the scorer (per-release FOMC minutes ≈ document).

Outputs:
- ``docs/verification/lexicon_validation.md`` — human-readable report
- ``docs/verification/lexicon_validation_confusion_matrix.csv`` — raw matrix
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


# Make the regime_detection package importable when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from regime_detection.central_bank_text import score_text  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOG = logging.getLogger(__name__)


_LABEL_NAMES = {0: "dovish", 1: "hawkish", 2: "neutral"}


def _map_net_score_to_label(net_score: float) -> int:
    """Map lexicon net_score in [-1, +1] to the corpus's 3-class label."""
    if pd.isna(net_score):
        return 2  # no lexicon hits → neutral by default
    if net_score > 0:
        return 1  # hawkish
    if net_score < 0:
        return 0  # dovish
    return 2  # exact tie → neutral


def score_sentence_level(df: pd.DataFrame) -> dict[str, object]:
    """Score each labeled sentence individually."""
    predicted: list[int] = []
    hit_mask: list[bool] = []
    for sentence in df["sentence"]:
        score = score_text(sentence)
        predicted.append(_map_net_score_to_label(score.net_score))
        hit_mask.append(score.hawkish_count + score.dovish_count > 0)

    y_true = df["label"].to_numpy()
    y_pred = np.array(predicted)
    coverage = float(np.mean(hit_mask))
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[_LABEL_NAMES[i] for i in [0, 1, 2]],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "n": int(len(df)),
        "coverage": coverage,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            _LABEL_NAMES[i]: {
                "precision": float(report[_LABEL_NAMES[i]]["precision"]),
                "recall": float(report[_LABEL_NAMES[i]]["recall"]),
                "f1": float(report[_LABEL_NAMES[i]]["f1-score"]),
                "support": int(report[_LABEL_NAMES[i]]["support"]),
            }
            for i in [0, 1, 2]
        },
        "confusion_matrix": cm.tolist(),
        "hit_breakdown_by_true_label": {
            _LABEL_NAMES[i]: float(
                np.mean([hit_mask[j] for j, v in enumerate(y_true) if v == i])
            )
            for i in [0, 1, 2]
        },
    }


def score_document_pooled(df: pd.DataFrame) -> dict[str, object]:
    """Pool sentences by (year, modal_label) and score the pooled text.

    For each year, choose the modal label (the most-common labeled
    class), concatenate all sentences with that label, and score the
    pooled paragraph. This mimics document-level scoring of a single
    annual policy direction.
    """
    rows: list[dict[str, object]] = []
    for year, subset in df.groupby("year"):
        modal_label = int(subset["label"].mode().iloc[0])
        pooled_text = " ".join(subset[subset["label"] == modal_label]["sentence"].tolist())
        score = score_text(pooled_text)
        rows.append(
            {
                "year": int(year),
                "n_sentences": int(len(subset)),
                "modal_label": modal_label,
                "modal_label_name": _LABEL_NAMES[modal_label],
                "hawkish_count": score.hawkish_count,
                "dovish_count": score.dovish_count,
                "total_tokens": score.total_tokens,
                "net_score": score.net_score,
                "predicted_label": _map_net_score_to_label(score.net_score),
                "predicted_label_name": _LABEL_NAMES[_map_net_score_to_label(score.net_score)],
            }
        )
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    correct = (out["predicted_label"] == out["modal_label"]).sum()
    return {
        "n_years": int(len(out)),
        "correct": int(correct),
        "accuracy": float(correct / len(out)) if len(out) else 0.0,
        "rows": out,
    }


def write_report(
    *,
    sentence_metrics: dict[str, object],
    document_metrics: dict[str, object],
    out_path: Path,
    cm_csv_path: Path,
) -> None:
    cm = np.array(sentence_metrics["confusion_matrix"])
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{_LABEL_NAMES[i]}" for i in [0, 1, 2]],
        columns=[f"pred_{_LABEL_NAMES[i]}" for i in [0, 1, 2]],
    )
    cm_df.to_csv(cm_csv_path)

    doc_rows = document_metrics["rows"]

    lines: list[str] = [
        "# v2 §2A Lexicon Validation — Trillion Dollar Words corpus",
        "",
        "Validates the deterministic hawkish/dovish lexicon in",
        "`src/regime_detection/central_bank_text.py` against the Shah,",
        "Paturi & Chava 2023 labeled FOMC corpus (EMNLP 2023;",
        "`gtfintechlab/fomc_communication`).",
        "",
        "Generated by `scripts/validate_central_bank_text_lexicon.py`.",
        "",
        "## 1. Sentence-level validation (strict)",
        "",
        f"- Sentences scored: **{sentence_metrics['n']}**",
        f"- Sentences with ≥1 lexicon hit (coverage): "
        f"**{sentence_metrics['coverage']:.1%}**",
        f"- Overall accuracy: **{sentence_metrics['accuracy']:.1%}**",
        f"- Macro F1: **{sentence_metrics['macro_f1']:.3f}**",
        f"- Weighted F1: **{sentence_metrics['weighted_f1']:.3f}**",
        "",
        "### Per-class precision / recall / F1",
        "",
        "| Class | Support | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for cls in ["dovish", "hawkish", "neutral"]:
        row = sentence_metrics["per_class"][cls]
        lines.append(
            f"| {cls} | {row['support']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Lexicon-hit rate by true label",
            "",
            "How often the lexicon fires *at all* on each true class. "
            "Reveals whether low recall is a vocabulary-gap problem "
            "(no hits → defaulted to neutral) or a sign-disagreement "
            "problem (hits exist but predict the wrong direction).",
            "",
            "| True label | Hit rate (lexicon fires) |",
            "|---|---|",
        ]
    )
    for cls in ["dovish", "hawkish", "neutral"]:
        rate = sentence_metrics["hit_breakdown_by_true_label"][cls]
        lines.append(f"| {cls} | {rate:.1%} |")
    lines.extend(
        [
            "",
            "### Confusion matrix",
            "",
            "Rows = true label, columns = predicted label.",
            "",
            cm_df.to_markdown(),
            "",
            f"Raw CSV: `{cm_csv_path.relative_to(REPO_ROOT)}`",
            "",
            "## 2. Document-level validation (pooled by year)",
            "",
            "Each year's modally-labeled sentences are concatenated into "
            "one pooled document and scored. Predicted label = sign of "
            "the pooled `net_score`; ties / no-hits → neutral. This "
            "approximates how the engine actually consumes the scorer "
            "(per-FOMC-minutes document, not per-sentence).",
            "",
            f"- Years covered: **{document_metrics['n_years']}**",
            f"- Correct (predicted = modal): **{document_metrics['correct']}**",
            f"- Document-level accuracy: **{document_metrics['accuracy']:.1%}**",
            "",
            "### Per-year detail",
            "",
            doc_rows[
                [
                    "year",
                    "n_sentences",
                    "modal_label_name",
                    "hawkish_count",
                    "dovish_count",
                    "net_score",
                    "predicted_label_name",
                ]
            ].to_markdown(index=False),
            "",
            "## 3. Honest read",
            "",
            "Sentence-level accuracy of a bag-of-keywords scorer against "
            "a single-sentence labeling task is the **most stringent** "
            "test the lexicon can face. Two failure modes show up here:",
            "",
            "- **Coverage gap.** Many sentences carry no lexicon term at "
            "all. Those default to neutral, which inflates neutral "
            "recall and depresses hawkish/dovish recall mechanically.",
            "- **Negation / conditional / dissent blindness.** The "
            "lexicon counts whole-word hits without clause scope. A "
            "sentence containing `not persistent inflation` still "
            "counts a hawkish hit.",
            "",
            "Document-level (pooled) is closer to the engine's actual "
            "use case. If document accuracy is materially higher than "
            "sentence accuracy, the lexicon is doing the right thing at "
            "the granularity the engine consumes it — and a sentence-"
            "level upgrade (e.g. `gtfintechlab/FOMC-RoBERTa`) would only "
            "add value if the engine moves to sentence-level evidence "
            "in the future.",
            "",
            "## 4. Decision criteria",
            "",
            "Per the audit follow-up plan in "
            "`docs/spec_code_data_audit_2026_05_15.md`:",
            "",
            "- **Document accuracy ≥ 70%** → lexicon is good enough for "
            "evidence-only use; defer the FinBERT/FOMC-RoBERTa upgrade.",
            "- **60% ≤ document accuracy < 70%** → marginal. Decide on "
            "deployment-cost grounds.",
            "- **Document accuracy < 60%** → upgrade to "
            "`gtfintechlab/FOMC-RoBERTa` is justified despite the "
            "dependency cost.",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    LOG.info("wrote report → %s", out_path)
    LOG.info("wrote confusion matrix → %s", cm_csv_path)


def main() -> int:
    LOG.info("loading gtfintechlab/fomc_communication ...")
    ds = load_dataset("gtfintechlab/fomc_communication")
    df = pd.concat(
        [ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True
    )
    LOG.info(
        "corpus loaded: %d sentences, years %d→%d",
        len(df),
        df["year"].min(),
        df["year"].max(),
    )
    label_dist = Counter(df["label"])
    LOG.info(
        "label distribution: dovish=%d hawkish=%d neutral=%d",
        label_dist[0],
        label_dist[1],
        label_dist[2],
    )

    LOG.info("scoring sentence-level ...")
    sentence_metrics = score_sentence_level(df)
    LOG.info(
        "sentence: accuracy=%.3f macro_f1=%.3f coverage=%.3f",
        sentence_metrics["accuracy"],
        sentence_metrics["macro_f1"],
        sentence_metrics["coverage"],
    )

    LOG.info("scoring document-level (pooled by year) ...")
    document_metrics = score_document_pooled(df)
    LOG.info(
        "document (pooled-by-year): accuracy=%.3f",
        document_metrics["accuracy"],
    )

    verification_dir = REPO_ROOT / "docs" / "verification"
    out_path = verification_dir / "lexicon_validation.md"
    cm_csv_path = verification_dir / "lexicon_validation_confusion_matrix.csv"
    write_report(
        sentence_metrics=sentence_metrics,
        document_metrics=document_metrics,
        out_path=out_path,
        cm_csv_path=cm_csv_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
