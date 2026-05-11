# TorchRobotics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Differentiable robot kinematics tree from URDF/MJCF formats, with differentiable planning objectives (obstacle avoidance, self-collision, end-effector tracking).

## Requirements

- Python >= 3.9
- PyTorch >= 2.0

## Installation

Activate your conda/Python environment and run:

```bash
pip install -e .
```

## Examples

Forward kinematics for all available robot models:

```bash
python examples/forward_kinematics.py
```

Inverse kinematics via Adam optimization:

```bash
python examples/inverse_kinematics.py
```

## Acknowledgements

Parts of the kinematics tree implementation are based on [differentiable-robot-model](https://github.com/facebookresearch/differentiable-robot-model) (Meta AI).

## Contact

- [An Le](https://www.ias.informatik.tu-darmstadt.de/Team/AnThaiLe) — [an@robot-learning.de](mailto:an@robot-learning.de)
- [Joao Carvalho](https://www.ias.informatik.tu-darmstadt.de/Team/JoaoCarvalho) — [joao@robot-learning.de](mailto:joao@robot-learning.de)

## Citation

If you found this repository useful, please consider citing:

```bibtex
@article{le2023accelerating,
  title={Accelerating motion planning via optimal transport},
  author={Le, An T and Chalvatzaki, Georgia and Biess, Armin and Peters, Jan R},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={78453--78482},
  year={2023}
}

@inproceedings{carvalho2023motion,
  title={Motion planning diffusion: Learning and planning of robot motions with diffusion models},
  author={Carvalho, Joao and Le, An T and Baierl, Mark and Koert, Dorothea and Peters, Jan},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1916--1923},
  year={2023},
  organization={IEEE}
}
```
