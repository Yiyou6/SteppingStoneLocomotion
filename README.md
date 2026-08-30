<div align="center">

# SF_TRON_FP 🤖

### Teaching a point-foot robot to *see* its way across a field of stepping stones — one blind step at a time.

**Reinforcement learning · NVIDIA Isaac Sim · PPO · Sim-to-Real**

</div>

---

## The Story 🐣

Every baby learns to walk before it learns to *look* where it's going — and so does our robot.

`SF_TRON_FP` is a three-stage curriculum that trains the **SF-TRON1A** point-foot humanoid to stride across rugged stepping-stone terrain. We start by letting the robot stumble around **completely blind**, relying only on what its own joints and IMU can feel. Then we crack its eyes open with a **depth camera**. Finally, we teach it to *guess* the things no real robot can ever measure — ground-contact forces, external pushes, true velocity — so that everything it learned in simulation survives the jump to a real machine.

The result is a locomotion policy that goes from *"falling over in the dark"* to *"confidently hopping across stones with vision"*, and finally to *"running on hardware without cheating."*

---

## TL;DR ⚡

| Stage | Script | What the robot gets | What it learns |
|-------|--------|---------------------|----------------|
| 1️⃣ **Blind Walk** | `Run_1.py` | Proprioception only (33-dim state) | A rock-solid base gait, no vision |
| 2️⃣ **Open Your Eyes** | `Run_2.py` | + Depth camera (231-dim state) | To *use* vision to navigate stones |
| 3️⃣ **No More Cheating** | `Run_with_Estimator.py` | + Privileged-state estimator | To recreate privileged info from history |

Every stage builds on the last. A frozen stage-1 "blind" policy even mentors stage-2 by contributing 20% of the action, so the robot never forgets how to walk while it learns to look.

---

## Features ✨

- 🧠 **PPO with GAE** — clean, self-contained Actor-Critic implementation with entropy bonus and action-smoothing loss.
- 👀 **Ray-cast depth camera** — a lightweight `18×11` depth map that costs a fraction of a rendering camera.
- 🪨 **Stepping-stone terrain** — procedurally generated height-field obstacles via Isaac Lab.
- 🌀 **Heavy domain randomization** — mass, center-of-mass, inertia, friction, restitution, PD gains, action delay, and random external pushes. The sim *tries* to knock your robot over.
- 🚀 **Massively parallel** — up to **4,000 agents** training simultaneously on a single GPU.
- 🎯 **Privileged-state estimation** — a separate network that reconstructs unmeasurable states from a rolling history, the secret sauce for sim-to-real.
- 🔁 **Sim-to-real pipeline** — ONNX export (`torch2onnx.py`) plus a ready-made deployment stack for the real robot.

---

## The Three Acts 🎬

### Act 1 — The Blindfold 🕶️ (`Run_1.py`)

The robot gets a 33-dimensional state: joint positions & velocities, body orientation, angular velocity, a phase clock, the last action, and a velocity command. The depth map is **zeroed out**.

It has no idea the ground is full of holes. It just learns to walk *really* well.

### Act 2 — The Glasses 👓 (`Run_2.py`)

The state explodes to 231 dimensions as the depth camera comes online. But we don't throw away what the robot learned — the frozen stage-1 policy keeps whispering in its ear:

```
action = 0.2 × blind_policy + 0.8 × camera_policy
```

A gentle hand-off from instinct to perception.

### Act 3 — The Magician 🎩 (`Run_with_Estimator.py`)

Here's the catch: real robots can't measure ground-contact force, external pushes, or their own true velocity in the wild. Yet the policy *used* that privileged info to train.

So we train an **Estimator** network to *reconstruct* those 8 privileged values from a 10-step rolling window of ordinary proprioception. The deployed policy walks with the estimator's guesses — no cheating required.

---

## How It Works ⚙️

```
┌────────────┐    state    ┌──────────────────┐   action   ┌─────────────┐
│  Isaac Sim │ ──────────▶ │   PPO Policy     │ ─────────▶ │  SF-TRON1A  │
│  (terrain, │             │  (Actor–Critic)  │            │   (8 motors)│
│  camera,   │ ◀────────── │                  │ ◀────────── │             │
│  sensors)  │  reward     └──────────────────┘   scaled   └─────────────┘
└────────────┘                                                  │
      ▲                                                         │
      └────────────────── domain randomization ◀────────────────┘
```

The reward function is a careful stew of ingredients: velocity tracking, body-height and orientation tracking, foot-constraint penalties, a single-support gait reward, air-time bonuses, and a firm `-10` penalty for face-planting. The robot learns, very quickly, that falling down is bad.

---

## Repo Structure 🗂️

```
SF_TRON_FP/
├── Run_1.py                    # Stage 1: blind walking
├── Run_2.py                    # Stage 2: camera-assisted walking
├── Run_with_Estimator.py       # Stage 3: privileged-state estimation
├── SRC/
│   ├── Config/                 # Environment / robot / PPO hyperparameters
│   ├── Env/                    # Isaac Sim environment, terrain, sensors
│   ├── PPO/                    # Actor-Critic network & replay buffer
│   ├── Estimator/              # Privileged-state estimator network
│   ├── Plotter/                # Real-time plotting helpers
│   └── Utils/                  # Transformations & math helpers
└── Model/
    ├── Robot_Model/            # SF_TRON1A USD + configs
    ├── NN_Model/               # Trained checkpoints (.pth) & ONNX export
    └── tron1-rl-deploy-python-main/   # Real-robot deployment stack
```

---

## Requirements 📦

| Dependency | Version |
|------------|---------|
| NVIDIA Isaac Sim | 5.1.0 |
| Isaac Lab | latest |
| PyTorch | ≥ 2.x |
| CUDA | 12.x (GPU required) |

> 🖥️ **Heads-up:** this project needs a CUDA-capable GPU and a working Isaac Sim installation. No GPU, no party.

---

## Getting Started 🚀

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/SF_TRON_FP.git
cd SF_TRON_FP

# 2. Fire up Isaac Sim / Isaac Lab (see their docs), then run a stage
python Run_1.py                 # Stage 1 — blind walking
python Run_2.py                 # Stage 2 — vision walking
python Run_with_Estimator.py    # Stage 3 — estimator + sim-to-real
```

**Train vs. play:** flip `train` in `SRC/Config/Config.py` (and `TS_Config.py` for stage 3). `True` trains and saves checkpoints under `Model/NN_Model/`; `False` loads the best model and renders for inspection.

---

## From Sim to Reality 🔌

The deployment stack under `Model/tron1-rl-deploy-python-main/` bridges the gap:

1. **Export** — convert trained policies to ONNX with `Model/NN_Model/torch2onnx.py`.
2. **Simulate** — sanity-check on the real robot's `mujoco` model.
3. **Deploy** — run the LimX SDK controller on hardware (`ROBOT_TYPE` + `RL_TYPE` env vars, robot IP as an argument).

The estimator-trained policy (Act 3) is the one built to survive the real world.

---

## Acknowledgements 🙏

Built on the shoulders of giants: [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim), [Isaac Lab](https://isaac-sim.robotics.ethz.ch/), and the **SF-TRON1A** platform by [LimX Dynamics](https://www.limxdynamics.com/).

---

<div align="center">

*Train hard, fall often, walk anyway.* 🦿

</div>
