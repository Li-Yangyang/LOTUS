#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:21:08 2021

@author: yangyangli
"""
import os

fp = os.path.dirname(os.path.realpath(__file__))

GCOG_NLTE_LIB = os.path.join(fp, "package_data/gcoglib/nlte_v0.tar.gz")
GCOG_LTE_LIB = os.path.join(fp, "package_data/gcoglib/lte_v0.tar.gz")

GCOG_NLTE_DIR = os.path.join(fp, "package_data/gcoglib/gcoglib_nlte_v0")
GCOG_LTE_DIR = os.path.join(fp, "package_data/gcoglib/gcoglib_lte_v0")

EWDIFF_LIB = os.path.join(fp, "package_data/ewdiff/ewdiff_v0.tar.gz")