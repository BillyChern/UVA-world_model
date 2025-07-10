# World-Model PPO on PushT

This directory contains the **minimal working prototype** demonstrating how to
train a reinforcement-learning policy *entirely inside* a pretrained UVA
video-and-action model, then evaluate (and optionally finetune) using a real
PushT environment.

---

## 1. Setup

```bash
# Activate the UVA conda env or create a new one
mamba env create -f conda_environment.yml   # if you have not installed UVA
conda activate uva

# Install extra requirements for this folder
pip install -r World-model/requirements.txt
```

GPU with CUDA-enabled PyTorch is strongly recommended.

---

## 2. Train PPO inside the world model

```bash
python World-model/train_policy.py \
  --checkpoint checkpoints/pusht.ckpt   # path to UVA PushT checkpoint
  --timesteps 1000000                   # adjust as needed
  --logdir runs/ppo_uva                 # tensorboard + policy files
```

• The script creates a `UVAWorldModelEnv` which serves as the simulator.
• Stable-Baselines3 PPO is used; logs are written to TensorBoard.

---

## 3. Evaluate the learned policy in the real PushT simulator

```bash
python World-model/eval_policy_real.py \
  --policy runs/ppo_uva/trained_policy.zip \
  --episodes 20
```

Returns for each episode and the mean score are printed.

---

## 4. Online finetuning of the world model (optional)

```bash
python World-model/finetune_uva.py \
  --checkpoint checkpoints/pusht.ckpt \
  --episodes 50         # collect data
  --epochs 5            # finetune steps on collected data
  --save finetuned_uva.ckpt
```

The script currently collects random-policy trajectories to demonstrate the
pipeline.  Swap in your PPO policy to gather higher-quality data and finetune.

---

## Files

| File | Purpose |
|---|---|
| `uva_world_env.py` | Wraps the UVA model as a Gym simulator. |
| `train_policy.py` | PPO training script using the world model. |
| `real_pusht_env.py` | Physics-based PushT implementation for evaluation. |
| `eval_policy_real.py` | Runs a trained policy in the real env. |
| `finetune_uva.py` | Prototype for online finetuning of UVA with new data. |
| `requirements.txt` | Additional Python dependencies. |

---

## Next Steps

* Replace random data collection in `finetune_uva.py` with policy rollouts.
* Move to real-robot datasets by swapping `RealPushTEnv` with your robotics
  interface.
* Experiment with larger UVA checkpoints or tasks beyond PushT. 