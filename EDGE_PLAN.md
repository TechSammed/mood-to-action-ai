# EDGE_PLAN.md — On-Device / Mobile Deployment

## Target: < 200ms latency, < 15MB total, fully offline

## Model compression

| Component | Current | Compressed | Method |
|---|---|---|---|
| SBERT encoder | 22 MB | ~6 MB | ONNX + INT8 quantisation |
| State model | ~2 MB | ~2 MB | XGBoost .ubj native |
| Intensity model | ~1 MB | ~1 MB | XGBoost .ubj native |
| Decision engine | ~10 KB | ~10 KB | Pure logic, no model |
| **Total** | **~25 MB** | **~9 MB** | |

## Latency budget (mid-range Android 2022)

| Step | Time |
|---|---|
| Text preprocessing | < 2ms |
| SBERT INT8 encoding | ~40ms |
| XGBoost state predict | ~3ms |
| XGBoost intensity predict | ~3ms |
| Uncertainty + decision | < 1ms |
| **Total** | **~50ms** |

## Platform deployment

### iOS
- Export to CoreML via `coremltools`
- ONNX → CoreML for transformer
- XGBoost → CoreML via `coremltools.converters`

### Android
- ONNX Runtime Android SDK
- XGBoost `.ubj` model via Java binding

### Export commands
```python
# XGBoost to ONNX
import xgboost as xgb
model.save_model("state_model.ubj")
# then: onnxmltools.convert_xgboost(model)

# SBERT to ONNX (quantised)
# optimum-cli export onnx --model all-MiniLM-L6-v2 --task feature-extraction model_onnx
```

## Privacy
All inference on-device. No journal text, metadata, or predictions leave the device.
This is critical for a mental health app — user reflections must never be transmitted.

## Tradeoffs

| Concern | Choice | Tradeoff |
|---|---|---|
| Quality vs size | INT8 SBERT | ~4% F1 drop, 3.5× smaller |
| Online vs offline | Fully offline | No personalisation, no privacy risk |
| Battery | Batch on wake | Slightly higher latency, much lower drain |

## Update strategy
- Model weights: ship with app updates
- Decision rules: remote config (no model retraining needed)
- Personalisation: local SQLite history + lightweight adapter fine-tuning