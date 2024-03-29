#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:21:08 2021

@author: yangyangli
"""
import os
from .download import default_download_dir, download_file, get_url

fp = os.path.dirname(os.path.realpath(__file__))
#print(fp)

#GCOG_NLTE_LIB = os.path.join(fp, "package_data/gcoglib/nlte_v0.tar.gz")
#GCOG_LTE_LIB = os.path.join(fp, "package_data/gcoglib/lte_v0.tar.gz")

GCOG_DIR = os.path.join(fp, "package_data/gcoglib/")
GCOG_NLTE_LIB = GCOG_DIR + "nlte_v0.tar.gz"
GCOG_LTE_LIB = GCOG_DIR + "lte_v0.tar.gz"
EWDIFF_DIR = os.path.join(fp, "package_data/ewdiff/")
EWDIFF_LIB = EWDIFF_DIR + "ewdiff_v0.tar.gz"


if not os.path.exists(GCOG_NLTE_LIB):
    GCOG_DIR = default_download_dir(GCOG_DIR)
    GCOG_NLTE_LIB = GCOG_DIR + "nlte_v0.tar.gz"
    url = get_url("nlte_v0.tar.gz")
    download_file(url, GCOG_DIR)

if not os.path.exists(GCOG_LTE_LIB):
    GCOG_DIR = default_download_dir(GCOG_DIR)
    GCOG_LTE_LIB = GCOG_DIR + "lte_v0.tar.gz"
    url = get_url("lte_v0.tar.gz")
    download_file(url, GCOG_DIR)

if not os.path.exists(EWDIFF_LIB):
    EWDIFF_DIR = default_download_dir(EWDIFF_DIR)
    EWDIFF_LIB = GCOG_DIR + "ewdiff_v0.tar.gz"
    url = get_url("ewdiff_v0.tar.gz")
    download_file(url, EWDIFF_DIR)

def fetch_EWLIB_MULTI():
    if not os.path.exists(os.path.join(fp, "package_data/EWLIB_largergrid2_v0.h5")):
        import time
        url = get_url(file_name="EWLIB_largergrid2_v0.h5", record_id="7474663")
        print("Downloading of the EW library will take around 20 minutes...")
        t1 = time.time();
        download_file(url, os.path.join(fp, "package_data/"))
        t2 = time.time();
        print("Downloading takes {0:.2f} mins".format((t2-t1) / 60.0))
    else:
        print("EW library has already existed: {0:s}".format(os.path.join(fp, "package_data/EWLIB_largergrid2_v0.h5")))
      
