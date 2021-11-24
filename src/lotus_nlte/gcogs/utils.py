#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:52:15 2021

@author: yangyangli
"""
import numpy as np
import h5py
import glob



def get_keys_and_atmos_pars(ewlibpath, stellar_type):
    """
    Get keys of the EW library for corresponding combinations of atmospheric 
    stellar parameters, given the type of star.

    Parameters
    ----------
    ewlibpath : str
        The path for the libary of EW, it must be a h5 file
    stellar_type : str
        The stellar type of your star, like:
            {spectral type, e.g. F, G, K}/{giant or subgiant or dwarf}/{metal_rich or metal_poor or very_metal_poor} or 
        the estimation of your atmospheric parameters in such form:
            {{T_low}_{T_high}/{logg_low}_{logg_high}/{feh_low}_{feh_high}}
            
    Returns
    -------
    keys : list of strs
        Keys of corresponding combinations of atmospheric 
        stellar parameters
    ini_cents : list
        the list of stellar parameters given the range defined by the stellar type
        the shape of the list is (N_selected_gridpoints, 4)

    """
    from ..utils import generate_ranges
    def visitor_func(name, node):
        if isinstance(node, h5py.Group):
            pass
        elif isinstance(node, h5py.Dataset): 
            ini_cent = [float("".join(c for c in node.name.split("/")[i] if not c.isalpha())) for i in range(-5,-1)]
            if all([lower[i] <= ini_cent[i] <= upper[i] for i in range(len(ini_cent))]):
                allkeys.append(node.name.replace("/table", ''))
                ini_cents.append(ini_cent)
    
    f = h5py.File(ewlibpath, 'r')
    allkeys = []
    ini_cents = []
    Teff_range, logg_range, feh_range = generate_ranges(stellar_type)
    upper = [Teff_range[1], logg_range[1], feh_range[1], np.inf]
    lower = [Teff_range[0], logg_range[0], feh_range[0], -np.inf]
    f.visititems(visitor_func)
    keys = allkeys
    ini_cents = ini_cents
    f.close()
    
    return keys, ini_cents

def find_closest_model(wl, ep, ele, search_path, interp_method, interpolation, tarfile=None):
    """
    Find closest interpolated model under designated direcctory

    Parameters
    ----------
    wl : float
        wavelength of line
    ep : float
        excitation potential of line
    ele : str
        "FeI" or "FeII"
    search_path : str
        designated directory
    interp_method: str
        "[2-5]" for multivariate polynomial interpolation
    interpolation: bool
        True: use interpolated GCOG
        False: get GCOG from EW library
    tarfile: tarfile.Tarfile or None
        if None, search closest model in a directory
        else search in a tarfile

    Returns
    -------
    list:
        length > 0 : closest model found;
        length = 0 : can't find closest model
        
    """
    
    for w in [wl-0.01, wl, wl+0.01]:
        for e in [ep-0.01, ep, ep+0.01]:
            line = format(w, ".2f") +"_"+ format(e, ".2f") +"_" + ele
            if interpolation:
                fname = glob.glob(search_path + line + "_" + interp_method + ".sav")
            else:
                names = tarfile.getnames()
                fname = list(filter(lambda x: search_path + line in x, names))
            if len(fname) > 0:
                return fname
    
    return fname
