![Logo](figures/banner.gif)
<h1 align="center">
  <b>iter-inla</b>
  <br>
</h1>

<h4 align="center">A framework for state and parameter estimation in nonlinear dynamical systems using iterated INLA.</h4>
<br>

## Overview

This repository contains the accompanying implementation for the UAI paper "Iterated INLA for state and parameter estimation in nonlinear dynamical systems".

## Dependencies

- NumPy
- SciPy
- Matplotlib
- scikit-learn
- [scikit-sparse fork with Takahashi equations](https://github.com/rafaelanderka/scikit-sparse)
- [findiff fork with periodic boundary conditions](https://github.com/rafaelanderka/findiff)


## Usage

The `demos` directory provides practical examples of the framework with interactive animations.
Additionally, multiple benchmarking scripts are provided in the `benchmarks` directory.

## Citing
If you found this useful, please consider citing:

```
@article{anderka2024iterated,
  title={Iterated {INLA} for State and Parameter Estimation in Nonlinear Dynamical Systems},
  author={Anderka, Rafael and Deisenroth, Marc Peter and Takao, So},
  journal={Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence},
  year={2024},
  publisher={PMLR}
}
```

## License
MIT
