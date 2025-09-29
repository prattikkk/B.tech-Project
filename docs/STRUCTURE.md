# Project Structure

This project is organized to keep code small and GitHub-friendly while storing heavy outputs in ignored `artifacts_*/` folders.

```
.
├─ Phase_One.py                 # Preprocessing pipeline
├─ Phase_Two.py                 # Model training, export, evaluation
├─ IDSAI.csv                    # Dataset (ignored by Git)
├─ artifacts_phase1/            # Outputs of Phase 1 (ignored)
├─ artifacts_phase2/            # Outputs of Phase 2 (ignored)
├─ artifacts_phase3/            # Outputs of Phase 3 runtime (logs, analysis, pointers)
├─ artifacts_phase4/            # Outputs of Phase 4 benchmarking/exports
├─ scripts/
│  ├─ run_phase1.ps1            # Helper script for Phase 1
│  └─ run_phase2.ps1            # Helper script for Phase 2
├─ configs/
│  ├─ optimization_config.example.json  # Template for pruning/FP16/static axis
│  └─ quant_config.example.json          # Template for quantization thresholds/strategy
├─ requirements.txt
├─ README.md
├─ Phase_Three.py               # MQTT subscriber / edge inference service
├─ Phase_Four.py                # Export, benchmarking, and MQTT runner
├─ Phase_Five.py                # Streamlit dashboard consuming predictions
├─ mqtt_publisher.py            # Test traffic generator (MQTT)
└─ docs/
   └─ STRUCTURE.md
```

## Artifacts (what they are used for)

- artifacts_phase1/
  - `data.npz` / `data.pkl`: tensors used for training.
  - `scaler.json` / `scaler.pkl`: feature standardization parameters.
  - `feature_order.json`: strict feature list/ordering.
  - `clip_bounds.json`: p1/p99 clipping bounds in raw space.
  - `class_weights.json`, `label_mapping.json`, `metadata.json`, etc.

- artifacts_phase2/
  - `final_model_hybrid.pth`: trained PyTorch model.
  - `model_hybrid.onnx`: ONNX export of the model.
  - `evaluation.json`: test/val metrics, thresholds, temperature.
  - `quantization_report.json`: post-training quantization summary.
  - `pruning_report.json`, `qat_report.json`: present when enabled via config.
  - `inference_wrapper.py`: light edge inference wrapper using ONNXRuntime if available.

- artifacts_phase3/
  - `phase3_predictions_log.csv`: rolling predictions log (subscriber mode).
  - `phase3_rolling_accuracy.png`, `phase3_confusion_matrix.png`: analysis artifacts.
  - `phase3_analysis_summary.json`: summary stats.
  - `current_model.txt`, `pending_model.txt`, `previous_model.txt`: model pointers for hot-swap.

- artifacts_phase4/
  - `phase4_benchmark.json`: benchmark summary for quantized or ONNX models.
