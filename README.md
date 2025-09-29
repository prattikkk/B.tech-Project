# IoT Anomaly Detection (Phase-wise)

Small, GitHub-friendly layout with phase scripts and reproducible artifacts.

## Structure

- `Phase_One.py` — data preprocessing: normalization, feature pruning, scaling, splits, and artifacts for training.
- `Phase_Two.py` — hybrid model training, calibration, (optional) pruning/quantization, ONNX export, and evaluation payloads.
- `IDSAI.csv` — dataset (ignored by Git to keep repo small).
- `artifacts_phase1/` — outputs from Phase 1 (ignored by Git).
- `artifacts_phase2/` — outputs from Phase 2 (ignored by Git).
- `requirements.txt` — runtime dependencies.
- `.gitignore` — excludes data, artifacts, and heavy binaries.
- `scripts/` — helper launchers for Windows PowerShell.
- `configs/` — example configs you can copy into `artifacts_phase2/` to enable pruning/QAT/ONNX axis options.

See `docs/STRUCTURE.md` for a visual tree and artifact details.

## Self-test (no external broker)

Run a single end-to-end self-test that simulates MQTT ingestion in-process, checks /metrics latency percentiles, backlog, and alert triggers:

```powershell
python .\integration_self_test.py
```

## Quickstart

1) Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Phase 1: preprocess dataset

```powershell
# Using helper script
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase1.ps1 -CsvPath .\IDSAI.csv -Artifacts .\artifacts_phase1

# Or direct
python .\Phase_One.py --csv .\IDSAI.csv --artifacts .\artifacts_phase1
```

3) Phase 2: train and export

```powershell
# Optional: copy config templates if you want pruning/QAT/static ONNX axis
Copy-Item .\configs\optimization_config.example.json .\artifacts_phase2\optimization_config.json -Force
Copy-Item .\configs\quant_config.example.json .\artifacts_phase2\quant_config.json -Force

# Using helper script
powershell -ExecutionPolicy Bypass -File .\scripts\run_phase2.ps1

# Or direct
python .\Phase_Two.py
```

If `onnxruntime` is installed, `Phase_Two.py` performs a small ONNX parity check and may export a quantized ONNX (`model_hybrid_q.onnx`) when the selected strategy is `all_linear`.

4) Phase 3: MQTT subscriber (edge inference service)

```powershell
# Run the subscriber with the best (float) model; set broker/port/topics as needed
python .\Phase_Four.py --run-mqtt --model .\artifacts_phase2\final_model_hybrid.pth --mqtt-broker localhost --mqtt-port 1883 --mqtt-topic iot/traffic --predictions-topic iot/traffic/predictions --health-topic iot/traffic/health

# Or quantize-on-load for faster inference (dynamic quantization)
python .\Phase_Four.py --run-mqtt --quantize-on-load
```

5) Phase 4: Benchmark and ONNX export

```powershell
# Benchmark quantized PyTorch (auto-quantizes if needed) and save results under artifacts_phase4
python .\Phase_Four.py --benchmark --num_samples 2000 --cpulimit 1

# Export ONNX and benchmark using onnxruntime
python .\Phase_Four.py --export-onnx
python .\Phase_Four.py --onnx-benchmark --num_samples 2000 --cpulimit 1
```

6) Phase 5: Live dashboard (Streamlit)

```powershell
# Start the Streamlit dashboard (listens to MQTT predictions and health topics)
streamlit run .\Phase_Five.py
```

7) Publisher (optional test traffic generator)

```powershell
# Publish dataset rows as MQTT messages to drive the subscriber/dashboard
python .\mqtt_publisher.py --base-dir . --split test --n-msgs 1000 --rate 20 --nested --unscale --protocol-as-name
```

## Pushing to GitHub

Initialize git (if not already), set the remote, and push:

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/prattikkk/B.tech-Project.git
git push -u origin main
```

## Notes

- The scripts auto-create `artifacts_phase1/` and `artifacts_phase2/` relative to the repo root.
- Large artifacts are git-ignored by default; commit only lightweight configs and code.
- For more control, edit `artifacts_phase2/optimization_config.json` and `artifacts_phase2/quant_config.json`.
