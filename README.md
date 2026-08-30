<div align="center">

# SF_TRON_FP 🤖

**Teach a point-foot robot to cross stepping stones — first blind, then with vision, then without cheating.**

*PPO · NVIDIA Isaac Sim · Sim-to-Real*

</div>

---

## What is this?

`SF_TRON_FP` trains the **SF-TRON1A** point-foot humanoid to walk across rugged stepping-stone terrain using Proximal Policy Optimization (PPO) in Isaac Sim. It's a three-stage curriculum, and each stage has a job:

| Stage | Script | What the robot sees | Goal |
|-------|--------|---------------------|------|
| 1️⃣ **Blind walk** | `Run_1.py` | Proprioception only (33-dim) | Rock-solid gait, no vision |
| 2️⃣ **With vision** | `Run_2.py` | + depth camera (231-dim) | Use vision to navigate stones |
| 3️⃣ **No cheating** | `Run_with_Estimator.py` | + privileged-state estimator | Rebuild unmeasurable states from history |

Stage 1 zeroes out the depth map and learns to walk by feel alone. Stage 2 freezes that "blind" policy and blends it with a new camera policy — `0.2 × blind + 0.8 × vision` — so the robot keeps its footing while learning to look. Stage 3 trains an **Estimator** to reconstruct privileged states (contact forces, external pushes, true velocity) from a rolling window of ordinary proprioception, which is what makes the policy deployable on real hardware.

---

## Features

- 🧠 **PPO + GAE** — self-contained Actor–Critic with entropy bonus and action-smoothing loss.
- 👀 **Ray-cast depth camera** — a cheap `18×11` depth map instead of a full render camera.
- 🪨 **Procedural stepping-stone terrain** — height-field obstacles via Isaac Lab.
- 🌀 **Heavy domain randomization** — mass, COM, inertia, friction, restitution, PD gains, action delay, and random pushes.
- 🚀 **Massively parallel** — up to **4,000 agents** on one GPU.
- 🔁 **Sim-to-real ready** — ONNX export + a deployment stack for the real robot.

---

## How it works

```
┌────────────┐    state    ┌──────────────────┐   action   ┌─────────────┐
│  Isaac Sim │ ──────────▶ │   PPO Policy     │ ─────────▶ │  SF-TRON1A  │
│ terrain /  │             │  (Actor–Critic)  │            │  (8 motors) │
│ camera /   │ ◀────────── │                  │ ◀────────── │             │
│ sensors    │  reward     └──────────────────┘  scaled    └─────────────┘
└────────────┘
```

The reward blends velocity tracking, body-height and orientation tracking, foot-constraint penalties, a single-support gait reward, air-time bonuses, and a hard `-10` for falling over.

---

## Repo structure

```
SF_TRON_FP/
├── Run_1.py                    # Stage 1: blind walking
├── Run_2.py                    # Stage 2: camera-assisted walking
├── Run_with_Estimator.py       # Stage 3: privileged-state estimation
├── SRC/
│   ├── Config/                 # Env / robot / PPO hyperparameters
│   ├── Env/                    # Isaac Sim scene, terrain, sensors
│   ├── PPO/                    # Actor-Critic network & replay buffer
│   ├── Estimator/              # Privileged-state estimator
│   ├── Plotter/                # Real-time plotting helpers
│   └── Utils/                  # Transformations & math helpers
└── Model/
    ├── Robot_Model/            # SF_TRON1A USD + configs
    ├── NN_Model/               # Checkpoints (.pth) & ONNX export
    └── tron1-rl-deploy-python-main/   # Real-robot deployment stack
```

---

## Requirements

| Dependency | Version |
|------------|---------|
| NVIDIA Isaac Sim | 5.1.0 |
| Isaac Lab | latest |
| PyTorch | ≥ 2.x |
| CUDA | 12.x (GPU required) |

> ⚠️ Needs a CUDA GPU and a working Isaac Sim install.

---

## Getting started

```bash
git clone https://github.com/<your-username>/SF_TRON_FP.git
cd SF_TRON_FP

python Run_1.py                 # Stage 1
python Run_2.py                 # Stage 2
python Run_with_Estimator.py    # Stage 3
```

Set `train` in `SRC/Config/Config.py` (and `TS_Config.py` for Stage 3): `True` trains and saves checkpoints to `Model/NN_Model/`; `False` loads the best model and renders for inspection.

---

## Sim-to-real

1. **Export** policies to ONNX with `Model/NN_Model/torch2onnx.py`.
2. **Validate** on the robot's `mujoco` model.
3. **Deploy** via the LimX SDK controller (`ROBOT_TYPE`, `RL_TYPE` env vars, robot IP as argument).

The Stage-3 estimator policy is the one built to survive reality.

---

## Acknowledgements

Built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim), [Isaac Lab](https://isaac-sim.robotics.ethz.ch/), and the **SF-TRON1A** platform by [LimX Dynamics](https://www.limxdynamics.com/).

<div align="center">

*Train hard, fall often, walk anyway.* 🦿

</div>
