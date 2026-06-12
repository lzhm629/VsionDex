# VisionDex: Real-Time Vision-Based Teleoperation for Dexterous Hand Simulation

VisionDex enables real-time vision-based teleoperation of a dexterous hand in MuJoCo.


## Installation

```bash
./setup.bash

conda activate visiondex
```


## Usage

### Perception

```bash
cd perception
python vision.py #开启视觉
```

### Simulation

```bash
cd simulation
python mujoco_simulation.py #另开终端，开启仿真控制
```