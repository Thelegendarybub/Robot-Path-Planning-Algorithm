# Autonomous Signal-Seeking Robot Navigation

A 2D autonomous robot navigation simulator built in Python, implementing gradient-based signal localization with reactive obstacle avoidance. The robot navigates an unknown environment to find the strongest signal region without a pre-built map.

---

## Problem Statement

Most beginner navigation implementations do one of two things: chase a gradient, or avoid obstacles. The real challenge is handling the conflict between them — what happens when the signal gradient points directly into blocked geometry, or when the robot gets trapped in a concave obstacle region?

This project addresses that conflict explicitly through a state-machine architecture that prioritizes geometry over optimization when necessary.

---

## How It Works

### Signal Field
The environment contains a hidden signal source. The robot cannot see it directly — it can only take noisy local measurements and estimate the gradient direction using finite differences.

### Three-Mode State Machine

| Mode | Trigger | Behavior |
|------|---------|----------|
| `NORMAL` | Path to signal is clear | Follow signal gradient directly |
| `AVOID` | Obstacle nearby, lateral risk | Blend signal gradient with tangent dodge |
| `BUG` | Corridor to signal blocked | Commit to wall-following until path clears |

The key insight: when trapped, the signal is no longer a reliable guide. The robot switches to geometry-first behavior (wall-following) and resumes gradient ascent only when the path forward is clear.

### Sensor Model
A simulated 360° range sensor casts rays across the environment. The robot uses these rays to:
- Detect corridor blockages via swept-width lookahead
- Track the nearest obstacle surface for wall-following
- Compute tangent directions for boundary navigation

### Wall Following (BUG Mode)
Tangent-style boundary following with closed-loop distance regulation:

```
direction = tangent + λ · error · normal_away
```

Where `error = following_distance - closest_distance`. This keeps the robot at a stable offset from the wall while progressing along its contour — similar in structure to Tangent Bug algorithms from the motion planning literature.

---

## Architecture

```
World / SignalField / RangeSensor
        ↓ (sensor data only, no direct geometry access)
Robot (state machine + control logic)
        ↓
Position update
```

The robot brain never directly accesses obstacle geometry — it only receives ray distances and directions from the sensor. This enforces a clean separation between world knowledge and robot perception, mirroring real hardware constraints.

---

## Results

The robot successfully navigates:
- Long walls and L-shaped obstacles
- Concave trap scenarios
- Noisy signal fields with gradient estimation uncertainty

Failure modes are understood: probe-point validity near boundaries, discrete timestep overshoot at high speeds, and cluster-averaging ambiguity at concave corners.

---

## Tech Stack

- **Python 3**
- **NumPy** — vector math, gradient estimation, ray casting
- **Matplotlib** — simulation visualization and mode history plotting

---

## Running the Simulation

```bash
git clone https://github.com/Thelegendarybub/Robot-Path-Planning-Algorithm
cd Robot-Path-Planning-Algorithm
pip install numpy matplotlib
python3 main.py
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `following_distance` | 0.35 | Target wall offset in BUG mode |
| `lambda_gain` | 2.0 | Wall-distance correction strength |
| `corridor_length` | 0.85 | Lookahead distance for blockage detection |
| `avoid_activation_distance` | 0.9 | Range at which AVOID mode triggers |
| `cluster_tolerance` | 0.18 | Ray averaging window for wall tracking |

---

## Concepts

- Reactive motion planning
- Tangent Bug algorithms
- Finite-difference gradient estimation
- Closed-loop distance regulation
- State machine architecture for autonomous systems

---

## ROS2 Extension

Navigation logic has been extended into a ROS2 (Humble) node architecture. The state machine maps directly onto a publisher/subscriber model:
- Sensor data → `/scan` topic
- Velocity commands → `/cmd_vel` topic
- Robot state managed via timer-driven control loop

*ROS2 node implementation in progress.*
