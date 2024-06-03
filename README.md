![Logo](figures/banner.gif)
<h1 align="center">
  <b>iter-inla</b>
  <br>
</h1>

<h4 align="center">A framework for state and parameter estimation in nonlinear dynamical systems using iterated INLA.</h4>
<br>

## Overview

This repository contains the accompanying implementation for the UAI paper "[Iterated INLA for State and Parameter Estimation in Nonlinear Dynamical Systems](https://arxiv.org/abs/2402.17036)".

## Dependencies

- NumPy
- SciPy
- Matplotlib
- scikit-learn
- [scikit-sparse fork with Takahashi equations](https://github.com/rafaelanderka/scikit-sparse)
- [findiff fork with periodic boundary conditions](https://github.com/rafaelanderka/findiff)

# Installation

This Python module depends on the suite-sparse library, which can be installed with your package manager of choice. For example, with `conda` we can install suite-sparse via:

```
# Install suite-sparse with conda
conda install -c conda-forge suitesparse
```

We also need custom forks of the `scikit-sparse` and `findiff` packages, which can be installed via:
```
# Install scikit-sparse fork
git clone https://github.com/rafaelanderka/scikit-sparse.git
pip install ./scikit-sparse
```
and
```
# Install findiff fork
git clone https://github.com/rafaelanderka/findiff.git
pip install ./findiff
```

Finally, we can install `iter-inla` with 
```
# Install iter-inla
git clone https://github.com/rafaelanderka/iter-inla.git
pip install ./iter-inla
```

The module can then be imported in Python as `iinla`.

## Usage

To get started, please have a look at the `demos` directory, which provides examples with live preview plots.

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
