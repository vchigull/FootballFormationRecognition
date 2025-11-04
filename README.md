# All-22 Formation Classifier (Starter)

This is a minimal, *working* starter repo for classifying **offensive** and **defensive** formations from a single All-22 image.

## What's included
- PyTorch ResNet50 fine-tuning pipeline
- Separate trainers for offense and defense
- Inference script to predict both labels on a single image
- Dummy dataset so the scripts **run end-to-end out of the box**
- CPU-only friendly (CUDA optional)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train offense (dummy data)
python src/train_offense.py --epochs 1 --bs 4

# Train defense (dummy data)
python src/train_defense.py --epochs 1 --bs 4

# Inference
python src/infer.py --image data/sample_all22.jpg   --offense_ckpt checkpoints/best_offense.pt   --defense_ckpt checkpoints/best_defense.pt
```
