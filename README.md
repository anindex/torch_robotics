# torch_kinematics_tree

This library implements Stochastic Gaussian Process Motion Planning algorithm in PyTorch. In Imitation Learning, we use this planner to sample trajectories from the learned Energy-Based Models encoding expert trajectories.

## Installation

Simply do

```python
pip install -e .
```

## Examples

This script do benchmark on computation time for all available robot kinematics

```azure
python examples/forward_kinematics.py
```

## Acknowledgements

A part of this implementation is inspired from the library [differentiable robot model](https://github.com/facebookresearch/differentiable-robot-model).

## Contact

If you have any questions or find any bugs, please let me know: [An Le](https://www.ias.informatik.tu-darmstadt.de/Team/AnThaiLe) an[at]robot-learning[dot]de