
## Materials

### GSM 2016 repository

https://github.com/jeffzhen/gsm2016

### ArXiv article

https://arxiv.org/abs/1605.04920

### GSM 2008, article

https://onlinelibrary.wiley.com/doi/fulpl/10.1111/j.1365-2966.2008.13376.x

## How to use

These procedures is checked on Ubuntu 20.04.

First, install Global Diffuse Sky Map library and some other dependencies:

```
pip install numpy scipy pandas matplotlib tqdm astropy astropy-healpix \
    git+https://github.com/telegraphic/pygdsm \
    git+https://github.com/taishi-hashimoto/python-antarrlib.git
```