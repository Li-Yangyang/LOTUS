#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:12:58 2021

@author: yangyangli
"""

import numpy as np
import pandas as pd
import h5py

import warnings

from .utils import get_keys_and_atmos_pars, find_closest_model
from ..config import GCOG_LTE_LIB, GCOG_NLTE_LIB

class SingleGCOG:
    """
    Base class for General Curve of Growth (GCOG)

    This intends for assemble discrete relationship between theoretical EW with
    other atsmopheric stellar parameters.

    Parameters
    ----------
    wavelength: float
        Target line wavelength
    ep: float
        Target line excitation potential , in unit of ev
    element: str
        "FeI" or "FeII"
    stellar_type: str:
        The stellar type of your star, like:
            {spectral type, e.g. F, G, K}/{giant or subgiant or dwarf}/{metal_rich or metal_poor or very_metal_poor}
        or
        the estimation of your atmospheric parameters in such form:
            {{T_low}_{T_high}/{logg_low}_{logg_high}/{feh_low}_{feh_high}}
    ewlibpath: str
        The path for the libary of EW, it must be a h5 file
    cal: str
        Types of EW, e.g. "lte" or "nlte"

    Methods
    -------
    assemble_hyper_surface:
        Collect EW and other stellar parameters from library and assemble them
        into an array as a shape of (N_selected_gridpoints, 5)
    """

    __slots__ = ["wavelength", "ep", "element", "stellar_type", "interpolated",
                 "ewlibpath", "interptar", "cal", "_keys", "_atmos_pars", "_hyper_surface"]

    def __init__(self, wavelength, ep, element, stellar_type, cal, interpolated=False,
     ewlibpath=None, keys=None, atmos_pars=None):
        self.wavelength = wavelength
        self.ep = ep
        self.element = element
        self.stellar_type = stellar_type
        self.cal = cal
        self.interpolated = interpolated
        if interpolated:
            import tarfile
            if self.cal == "lte":
                self.interptar = tarfile.open(GCOG_LTE_LIB)
            else:
                self.interptar = tarfile.open(GCOG_NLTE_LIB)
        else:
            if not ewlibpath:
                raise ValueError("Please assign your EW library!")
            self.ewlibpath = ewlibpath
            if keys==None or atmos_pars==None:
                self._keys, self._atmos_pars = get_keys_and_atmos_pars(ewlibpath, stellar_type)
            else:
                self._keys = keys
                self._atmos_pars = atmos_pars
            self._hyper_surface = None

    def load_model(self):
        if not self.interpolated:
            raise ValueError("Your model hasn't been interpolated yet")
        import joblib
        working_dir = self.interptar.getnames()[0] + "/" + self.stellar_type + "/"
        fname = find_closest_model(self.wavelength, self.ep,
                                    self.element, working_dir,
                                    "*", False,
                                    self.interptar)
        m = joblib.load(self.interptar.extractfile(fname[0]))
        return m

    def plot_interpolated_cog(self, teff, logg, vt, ews=None):
        if not self.interpolated:
            raise ValueError("Your model hasn't been interpolated yet")
        import matplotlib.pyplot as plt
        if not ews:
            ews = np.arange(1, 100, 0.1)
        generated_mets = [self.load_model().predict([[teff, logg, vt, ew]])[0] for ew in ews]
        fig = plt.plot(ews, generated_mets)
        plt.xlabel("EW(m$\AA$)")
        plt.ylabel("A(Fe)")
        return fig


    def assemble_hyper_surface(self):
        """
        Collect EW and other stellar parameters from library and assemble them
        into an array

        Returns
        -------
        ndarray
            An array with a shape of (N, 5), N is the number of grid points in
            your targerted region of library, this depends on the parameter of
            stellar_type

        """
        def get_row_no(k0):
            hdf = pd.HDFStore(self.ewlibpath, 'r')
            hdf0 = hdf.get(k0)
            idx = np.where((np.abs(hdf0.th_wavelength-self.wavelength)<=0.025)
                              & (np.abs(hdf0.th_EP - self.ep)<=0.02)
                             & (hdf0.element == self.element))[0]
            if idx.size!=0:
               idx = idx[0]
            else:
               idx = -1
            hdf.close()
            return idx

        if self.interpolated:
            raise NotImplementedError("Interpolated model doesn't have such method!")

        row_no = get_row_no(self._keys[0])
        if row_no == -1:
            warnings.warn("Data for interpolation is not enough!")
            self._hyper_surface = None
            return self._hyper_surface
        else:
            f = h5py.File(self.ewlibpath, 'r')
            if self.cal == "nlte":
                ews = [np.array(f[k+"/table"])[row_no][1][3] for k in self._keys]
            else:
                ews = [np.array(f[k+"/table"])[row_no][1][2] for k in self._keys]
            f.close()


            datapoints = np.concatenate((np.array(self._atmos_pars), np.transpose([ews])), axis=1)
            datapoints = datapoints[~np.isnan(datapoints).any(axis=1)]

            if datapoints.shape[0] <= 3:
                warnings.warn("Data for interpolation is not enough!")
                self._hyper_surface = None
                del datapoints
                return self._hyper_surface
            else:
                self._hyper_surface = datapoints
                print("Grid is prepared!")
                del datapoints
                return self._hyper_surface
