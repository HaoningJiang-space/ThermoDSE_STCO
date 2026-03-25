# ThermoDSE

A thermal-aware and comprehensive design space exploration framework for chiplet-based DNN accelerators. ThermoDSE integrates fine-grained task modeling with a uniform simulation and optimization framework that simultaneously considers chiplet granularity, NPU core granularity, task orchestration, and inter-chiplet communication under strict thermal and area constraints. (submitted to Transaction on Computer)

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
<!-- - [License](#license) -->

## Installation

### Package Requirements

1. gymnasium 1.1.1
2. botorch 0.14.0

Install via conda:

```bash
conda install gymnasium
conda install botorch -c gpytorch -c conda-forge
```

## Project Structure

- **core/**: Hardware-related files including library, task scheduling, and hardware model
- **nns/**: Neural network workloads
- **rl_opt/**: Baseline algorithms (Simulated Annealing, Reinforcement Learning)
- **tools/**: TED + SCBO (our method) and other scripts
- **tmp_x/**: Hotspot simulation directory template, for multiple processing, it will copy the template for each sub-process

## Usage

Before running experiments, make sure all dependencies are installed and the HotSpot path is configured.

### Run SCBO Algorithm

```bash
cd tools
python scbo_search.py -hp /HOTSPOT_PATH -maxA 300 -maxT 348 -sp ../tmp
```

### Run Baseline 1 (RL PPO)

```bash
python rl_ppo.py -b1 1 -hp /HOTSPOT_PATH -maxA 300 -maxT 348 -sp ../tmp_0
```

### Run Baseline 2 (TESA with Ideal Scheduling)

```bash
cd rl_opt
python sa_opt.py -b2 1 -hp /HOTSPOT_PATH -maxA 300 -maxT 348 -sp ../tmp_1
```

### Run Baseline 3 (TESA with No-Ideal Scheduling)

```bash
cd rl_opt
python sa_opt.py -b3 1 -hp /HOTSPOT_PATH -maxA 300 -maxT 348 -sp ../tmp_2
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

<!-- ## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->
