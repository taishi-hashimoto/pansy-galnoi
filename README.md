# pansy-galnoi

Galactic noise level estimator for antenna arrays.  
[The PANSY radar](https://pansy.eps.s.u-tokyo.ac.jp/en/index.html) is the original target for this program, but this is applicable to any antenna arrays.

## How it works

This program computes a product with the diffuse Galactic radio emission and the theoretical antenna pattern of an antenna array.

### The global diffuse sky models

- https://github.com/telegraphic/pygdsm

### Antenna array theory

- https://github.com/taishi-hashimoto/python-antarrlib

## Installation

Use pip.

```
pip install git+https://github.com/taishi-hashimoto/pansy-galnoi.git
```