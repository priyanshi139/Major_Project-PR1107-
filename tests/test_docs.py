import importlib.util, json
from pathlib import Path
from src.models.train_sepsis import run_training
from src.preprocessing.sepsis_preprocessing import patient_wise_split
from tests.test_sepsis_phase3 import synth

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ged", ROOT / "scripts" / "generate_eval_doc.py")
ged = importlib.util.module_from_spec(spec); spec.loader.exec_module(ged)


def test_eval_doc_uses_report_numbers_only(tmp_path):
    df = synth(); sp = patient_wise_split(df, seed=1)
    rep = run_training(df, sp, tmp_path / "a", tmp_path / "r", seed=1)
    md = ged.build_markdown(json.loads((tmp_path / "r" / "sepsis_metrics.json").read_text()))
    t = rep["results"]["logistic_regression"]["test"]
    assert f"{t['auroc']:.3f}" in md and "not clinically validated" in md and "Random AUPRC" in md


def test_required_docs_exist():
    for f in ["system_architecture", "data_dictionary", "methodology", "limitations_and_ethics",
              "demo_script", "research_paper_outline", "future_work", "PROJECT_STATUS"]:
        assert (ROOT / "docs" / f"{f}.md").stat().st_size > 200, f
