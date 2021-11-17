#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from . import utils
from . import interpolation
from . import gcogs
from . import optimize 
from . import sampling
from . import plot
#from .LOTUS_version import version as __version__

__author__ = "Yangyang Li"
__email__ = "ly95astro@gmail.com"
__uri__ = "https://github.com/Li-Yangyang/LOTUs"
__license__ = "BSD"
__description__ = "Determine atmospheric stellar parameters in non-LTE"
__copyright__ = "Copyright 2021 Yangyang Li"
__contributors__ = "https://github.com/Li-Yangyang/LOTUs/graphs/contributors"
__bibtex__ = __citation__ = """TBD"""
__version__="0.1"

__all__ = ["__version__", "gcogs", "optimize", "interpolation", "utils", "sampling", "plot"]