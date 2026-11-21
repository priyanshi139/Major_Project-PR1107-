"""Generate docs/model_evaluation.md from reports/sepsis_metrics.json (no numbers are typed by hand)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_path


def build_markdown(rep: dict) -> str:
    m = rep["meta"]
    L = ["# Model Evaluation - Sepsis module", "",
         "> Generated automatically from `reports/sepsis_metrics.json`. Academic prototype; not clinically validated.", "",
         f"- Selected model: **{m['selected_model']}** ({m['selection_rule']})",
         f"- Threshold: {m['threshold']:.4f} (maximum F1 on the validation set)",
         f"- Seed: {m['seed']} | patients per split: {m['split_patients']}",
         f"- Package versions: {m['versions']}", "",
         "## Test-set results (row-level, hourly rows)", "",
         "| Model | AUROC | AUPRC | Random AUPRC | Precision | Recall | F1 | Brier |", "|---|---|---|---|---|---|---|---|"]
    for n, r in rep["results"].items():
        t = r["test"]
        L.append(f"| {n} | {t['auroc']:.3f} | {t['auprc']:.3f} | {t['auprc_random_baseline']:.3f} | "
                 f"{t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | {t['brier']:.3f} |")
    L += ["", "## Confusion matrices (test, at the validation-chosen threshold)", "",
          "| Model | TN | FP | FN | TP |", "|---|---|---|---|---|"]
    for n, r in rep["results"].items():
        c = r["test"]["confusion_matrix"]
        L.append(f"| {n} | {c['tn']} | {c['fp']} | {c['fn']} | {c['tp']} |")
    L += ["", "## Validation-set results (used for selection and thresholding)", "",
          "| Model | AUROC | AUPRC | Threshold |", "|---|---|---|---|"]
    for n, r in rep["results"].items():
        v = r["val"]
        L.append(f"| {n} | {v['auroc']:.3f} | {v['auprc']:.3f} | {v['threshold']:.4f} |")
    L += ["", "## How to read this", "",
          "- **AUPRC vs the random baseline** is the headline comparison, because positives are rare. The random baseline equals the positive-row rate.",
          "- **Precision** tells how many flagged hours are labelled positive; low precision means many false alarms. **Recall** tells how many positive hours are caught.",
          "- Lowering the threshold increases recall and decreases precision; the chosen threshold is a validation-set compromise, not a medical standard.",
          "- Scores come from class-weighted models: treat them as ranking scores, not probabilities. See calibration bins in the JSON and `reports/sepsis_curves.png`.",
          "- The test set was not used for model selection or thresholding. Test metrics are noisy because the test set holds a limited number of septic patients.",
          "- Row-level metrics only. Patient-level early-warning performance and the official challenge utility score were not computed.", "",
          "![curves](../reports/sepsis_curves.png)", ""]
    return "\n".join(L)


def main():
    cfg = load_config()
    p = resolve_path(cfg, "reports_dir") / "sepsis_metrics.json"
    if not p.exists():
        sys.exit("reports/sepsis_metrics.json not found. Run: python scripts/train_sepsis.py")
    out = Path(__file__).resolve().parents[1] / "docs" / "model_evaluation.md"
    out.write_text(build_markdown(json.loads(p.read_text())), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
