# SENSE-X: Multi-Sensor Obstacle-Aware Aerial Reconnaissance & Human Detection

SENSE-X is an advanced multi-sensor drone platform and software pipeline engineered for **Disaster Rescue** (detecting victims under heavy rubble) and **Defense / Security** (through-wall infiltration detection in tunnels & buildings).

## Project Overview

- **Dual-Mode Operation:** Instant software toggle between **Disaster Rescue Mode** and **Defense/Security Mode**.
- **Payload Integration:** 
  - UWB Micro-Doppler Radar (Through-wall respiration/vital sign detection).
  - LWIR Radiometric Thermal Camera (Body heat detection).
  - Directional Acoustic Sensor Array + Audio Ego-Noise Cancellation.
  - RGB Camera + Real-time Edge AI Vision (YOLOv8-nano).
- **Obstacle-Aware Sensor Fusion (OASF):** Dynamically adjusts sensor confidence weights based on real-time environmental obstacle density ($\alpha$).
- **Priority-Based Action Index:** Generates actionable spatial heatmaps and target location vectors for emergency commanders and tactical teams.

---

## Directory Structure

```
SENSE_X/
├── data/
│   ├── raw/ (rgb, thermal, radar, audio)
│   ├── processed/ (fused telemetry)
│   └── metadata/
├── models/ (rgb, thermal, radar, audio models)
├── src/
│   ├── sensors/ (RGB, Thermal, Radar, Audio pipelines)
│   ├── fusion/ (Obstacle-Aware Sensor Fusion Engine)
│   ├── localization/ (Spatial target heatmaps)
│   ├── mission/ (Disaster & Defense workflow logic)
│   └── api/ (Command dashboard & alerts API)
├── scripts/ (Data acquisition & inspection)
├── configs/ (System parameter configs)
├── tests/ (Unit and integration tests)
├── demo/ (Interactive simulation UI)
└── docs/ (Phase 1 Proposal & Architectural Docs)
```

---

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Inspect Audio Dataset (HuggingFace DroneAudioSet):**
   ```bash
   python scripts/inspect_drone_audio.py
   ```

3. **Run Sensor Fusion Simulation:**
   ```bash
   python -m src.fusion.oasf_engine
   ```

---

## Documentation
- [Phase 1 Proposal Document](docs/SENSE_X_Phase1_Proposal.md)
