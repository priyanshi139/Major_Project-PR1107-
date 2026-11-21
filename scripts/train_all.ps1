# Train everything that needs data your laptop can download. Run from the project root with the venv active:
#   .\scripts\train_all.ps1                 # sepsis 3000 files per set (about 6000 patients)
#   .\scripts\train_all.ps1 -SepsisLimit 1000 -Quick     # smaller / faster check
param([int]$SepsisLimit = 3000, [switch]$Quick)
$ErrorActionPreference = "Stop"
function Step($name, $cmd) { Write-Host "`n=== $name ===" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "$name failed" } }

Step "Tabular diseases (diabetes, cardiovascular, hypertension)" "python scripts/download_tabular.py; python scripts/train_tabular.py"
Step "Sepsis: download PhysioNet/CinC 2019"  "python scripts/download_sepsis.py --limit $SepsisLimit --workers 8"
Step "Sepsis: prepare (patient-wise split)"  "python scripts/prepare_sepsis_data.py"
Step "Sepsis: ML model (logistic + gradient boosting)" "python scripts/train_sepsis.py"
$ep = if ($Quick) { "--epochs 2 --max-patients 300" } else { "--epochs 8 --batch-size 256" }
Step "Sepsis: causal Transformer + LSTM" "python scripts/train_causal_transformer.py $ep"
Step "Vital-sign forecasting vs persistence" "python scripts/train_forecast.py"
Step "ECG: download MIT-BIH + train" "python scripts/download_ecg.py; python scripts/train_ecg.py"
Step "Final check" "python scripts/check_environment.py"
Write-Host "`nDone. Start the app: streamlit run app.py" -ForegroundColor Green
