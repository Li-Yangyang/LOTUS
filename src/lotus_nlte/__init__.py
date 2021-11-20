#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__all__ = ["__version__", "gcogs", "optimize", "interpolation", "utils", "sampling", "plot"]
from . import utils
from .interpolation import *
from .gcogs.multigcogs import PolyMultiCGOG
from . import optimize
from . import sampling
from . import plot
#from .LOTUS_version import __version__
from .LOTUS_version import version as __version__

__author__ = "Yangyang Li"
__email__ = "ly95astro@gmail.com"
__url__ = "https://github.com/Li-Yangyang/LOTUs"
__license__ = "MIT"
__description__ = "Determine atmospheric stellar parameters in non-LTE"
__copyright__ = "Copyright 2021 Yangyang Li"
__contributors__ = "https://github.com/Li-Yangyang/LOTUs/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
#__version__ = "0.0.1a1"
