#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:12:32 2021

@author: yangyangli
"""
__all__ = [
    "SingleGCOG",
    "MultiGCOG",
    "PolyMultiGCOG"
]

from .gcog import SingleGCOG
from .multigcogs import MultiGCOG, PolyMultiGCOG
