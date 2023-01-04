<p align="center">
  <img width="20%" src="https://raw.githubusercontent.com/Li-Yangyang/LOTUS/main/doc/_static/logo.png">
  <br><br>
  <a href="http://lotus-nlte.readthedocs.io">
    <img src="https://readthedocs.org/projects/lotus_nlte/badge/?version=latest" alt="Docs">
  </a>
  <a href="https://github.com/Li-Yangyang/LOTUS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
  <a href="https://arxiv.org/abs/2207.09415">
    <img src="https://img.shields.io/badge/Arxiv-2207.09415-orange.svg" alt="Arxiv">
  </a>
</p>

# LOTUS
_LOTUS_ (non-LTE Optimization Tool Utilized for the derivation of atmospheric
Stellar parameters) is a python package for the derivation of stellar parameters via _Equivalent Width (EW)_ method with the assumption of
**1D Non Local Thermodynamic Equilibrium**. It mainly applies on the spectroscopic
data from high resolution spectral survey. It can provide extremely accurate
measurement of stellar parameters compared with non-spectroscipic analysis from
benchmark stars.

Full documentation at [lotus-nlte.readthedocs.io](https://lotus-nlte.readthedocs.io)

## Installation

The quickest way to get started is to use [pip](https://pip.pypa.io):

```bash
python -m pip install lotus-nlte==0.1.1
```
Notice that _LOTUS_ requires Python 3.7.*. You might create an independent environment to run this code.

## Usage

Check out the user guides and tutorial docs on [the docs
page](https://lotus-nlte.readthedocs.io) for details.

## Contributing

_LOTUS_ is an open source code so if you would like to contribute your work please
report an issue or clone this repository to your local end to contribute any changes.

## Attribution

Our paper has been submitted to _The Astronomical Journal_ and is being peer-reviewed. We also post it on arxiv and we will update citation after being accepted. If you use _LOTUS_ in your research, please cite:


    @ARTICLE{2022arXiv220709415L,
       author = {{Li}, Yangyang and {Ezzeddine}, Rana},
        title = "{LOTUS: A (non-)LTE Optimization Tool for Uniform derivation of Stellar atmospheric parameters}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2022,
        month = jul,
          eid = {arXiv:2207.09415},
        pages = {arXiv:2207.09415},
     archivePrefix = {arXiv},
       eprint = {2207.09415},
     primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220709415L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


