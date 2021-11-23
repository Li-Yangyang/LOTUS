#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:28:48 2020

@author: yangyangli
"""
import os, re, tarfile
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from scipy.optimize import curve_fit

import logging
import tqdm
logger = logging.getLogger(__name__)

def searchsorted2(known_array, test_array):
    """
    Search sort function to match the observation line list with modeled line list

    Parameters
    ----------
    known_array : 1D array
        This array contains those lines wavelength with EW larger than 1.0 in modeled
        line list.
    test_array : array_like
        This array contains those lines wavelength in observational line list.

    Returns
    -------
    indices : ndarray
        index of matched lines in modeled line .
    residual_abs : ndarray, size is same as indices
        the residual between each item in test_array and their found corresponding
        item in known_array.

    """
    index_sorted = np.argsort(known_array)
    known_array_sorted = known_array[index_sorted]
    known_array_middles = known_array_sorted[1:] - np.diff(known_array_sorted.astype('f'))/2
    idx1 = np.searchsorted(known_array_middles, test_array)
    indices = index_sorted[idx1]
    residual_abs = np.abs(test_array - known_array[indices])
    return indices, residual_abs

def find_outputfiles(tar, parameters=None):
    if parameters==None:
        outfile_index = []
        for i, member in enumerate(tar.getmembers()):
            if "out" in member.name:
                outfile_index.append(i)

        return outfile_index

def read_outputfile(filepath_or_buffer, skiprows, nrows, obs_linelist=None, level_model="./model_level/fe.v8.1"):
    #TODO:add read method for hierachical directory rather than tar file
    if skiprows == None and nrows == None:
        #Find # of start and end lines for EW data block
        no_start = []
        no_end = []
        for num, line in enumerate(filepath_or_buffer, 1):
            if b"KR" in line:
                no_start.append(num)
            if b"1 D LG TAUNY" in line:
                no_end.append(num)
        skiprows = no_start[1]+1
        nrows = no_end[1] - no_start[1] - 2
    else:
        skiprows=skiprows
        nrows=nrows
    #clean reduced output table
    atom_frame = pd.read_csv(
        filepath_or_buffer,
        delimiter=r"\s+",
        header=None,
        skiprows=skiprows,
        nrows=nrows,
        encoding='utf-8')
    #atom_fram.to_csv("output_atom.csv")
    #atom_frame = pd.read_csv("./new_output_atom", delimiter=r"\s+", header=None)
    atom_frame.loc[pd.isna(atom_frame[5]), 5] = np.nan
    atom_frame = atom_frame.apply(pd.to_numeric, errors='coerce')
    atom_frame.loc[atom_frame[5]<0, 5] = 0.0
    atom_frame.loc[atom_frame[4]<0, 4] = 0.0
    #atom_frame[5].astype = np.float


    # readin atom level file from ATOM and store index of FeI and FeII respectively
    atom_model = pd.read_csv(level_model, header=0, sep="\s+", index_col=False)

    idx_fei_th = np.where(np.logical_and(atom_frame[3].isin(atom_model[atom_model.ION==1].NK),\
                                         atom_frame[4] >= 0.0) == True)
    idx_feii_th = np.where(np.logical_and(atom_frame[3].isin(atom_model[atom_model.ION==2].NK),\
                                          atom_frame[4] >= 0.0) == True)


    if obs_linelist == None:

        #match with level model table
        merged1 = pd.merge(atom_model[atom_model.ION==1],\
                          atom_frame.iloc[idx_fei_th], left_on="NK", right_on=3, how="right")
        merged1 = merged1.sort_values(by=[1])
        merged1["E(ev)"] = np.array(merged1["E[cm-1]"]*1.23986e-4)

        merged2 = pd.merge(atom_model[atom_model.ION==2],\
                          atom_frame.iloc[idx_feii_th], left_on="NK", right_on=3, how="right")
        merged2 = merged2.sort_values(by=[1])
        merged2["E(ev)"] = np.array(merged2["E[cm-1]"]*1.23986e-4) - 7.9024

        #merge both tables
        final_th = pd.concat([merged1,
                              merged2])
        #calculate ew and ep of each line
        eqw_lte = np.array(final_th[4]/final_th[5]).round(3)
        eqw_nlte = np.array(final_th[4]).round(3)
        element = np.append(["FeI" for x in range(np.shape(idx_fei_th)[1])],\
                            ["FeII" for x in range(np.shape(idx_feii_th)[1])])

        d = {'element': element,
             'th_wavelength': np.array(final_th[1]),
             'th_EP': np.array(final_th["E(ev)"]),
             'lte_EW': eqw_lte,
             'nlte_EW': eqw_nlte}

        return pd.DataFrame(d)

    else:
        observation_frame = pd.read_csv(obs_linelist)
        idx_fei_ob = np.where((observation_frame["Line"] == "FeI"))
        idx_feii_ob = np.where((observation_frame["Line"] == "FeII"))

        #close-enough check for wavelength
        idx_closest1, residuals_abs1 = searchsorted2(np.array(atom_frame.iloc[idx_fei_th][1]), np.array(observation_frame.iloc[idx_fei_ob]["lambda"]))
        #search distance of wavelength is 0.025
        idx_closest_enough1 = residuals_abs1 <= 0.025
        idx_closest2, residuals_abs2 = searchsorted2(np.array(atom_frame.iloc[idx_feii_th][1]), np.array(observation_frame.iloc[idx_feii_ob]["lambda"]))
        idx_closest_enough2 = residuals_abs2 <= 0.025

        #match with level model table
        merged1 = pd.merge(atom_model[atom_model.ION==1],\
                          atom_frame.iloc[idx_fei_th].iloc[idx_closest1[idx_closest_enough1]],\
                              left_on="NK", right_on=3, how="right")
        merged1 = merged1.sort_values(by=[1])
        merged1["E(ev)"] = np.array(merged1["E[cm-1]"]*1.23986e-4)

        merged2 = pd.merge(atom_model[atom_model.ION==2],\
                          atom_frame.iloc[idx_feii_th].iloc[idx_closest2[idx_closest_enough2]],\
                              left_on="NK", right_on=3, how="right")
        merged2 = merged2.sort_values(by=[1])
        merged2["E(ev)"] = np.array(merged2["E[cm-1]"]*1.23986e-4) - 7.9024

        #merge both tables
        final_th = pd.concat([merged1,
                    merged2], ignore_index=True)

        final_ob = pd.concat([observation_frame.iloc[idx_fei_ob].iloc[idx_closest_enough1].sort_values(by=["lambda"], ascending=True),
                     observation_frame.iloc[idx_feii_ob].iloc[idx_closest_enough2]], ignore_index=True)

        #close-enough check for EP
        residuals_ep = np.abs(np.array(final_th["E(ev)"]) - np.array(final_ob["Ex.P"]))
        #search distance of EP is 0.02
        idx_closest_enough_ep = residuals_ep <= 0.02

        final_th = final_th.iloc[idx_closest_enough_ep]
        final_ob = final_ob.iloc[idx_closest_enough_ep]

        eqw_lte = np.array(final_th[4]/final_th[5]).round(3)
        eqw_nlte = np.array(final_th[4]).round(3)
        d = {'element': final_ob["Line"],
             'th_wavelength':  np.array(final_th[1]),
             'th_EP': np.array(final_th["E(ev)"]),
             'lte_EW': eqw_lte,
             'nlte_EW': eqw_nlte}
        return pd.DataFrame(d)

def spectralib(gridpath, grid_readme, obs_linelist="./linelist/final.linelist", level_model="./model_level/fe.v8.1", ewlibpath="EWLIB_grid2.h5"):
    ##get rid of naturalwarnings here
    #import warnings
    #import tables
    #original_warnings = list(warnings.filters)
    #warnings.simplefilter('ignore', tables.NaturalNameWarning)

    store = pd.HDFStore(ewlibpath)
    if gridpath.endswith(".gz"):
        tar = tarfile.open(gridpath)
        outfile_index = find_outputfiles(tar)

        for i in outfile_index:
            teff = tar.getmembers()[i].name.split("/")[4].split("_")[1] + "K"
            logg = "logg" + tar.getmembers()[i].name.split("/")[4].split("_")[2]
            z = "z" + tar.getmembers()[i].name.split("/")[4].split("_")[3]
            vt = "vt" + tar.getmembers()[i].name.split("/")[4].split("_")[4]
            try:
                store.append(gridpath+teff+"/"+logg+"/"+z+"/"+vt,
                         read_outputfile(tar.extractfile(tar.getmembers()[i]), skiprows=None, nrows=None, obs_linelist=obs_linelist, level_model=level_model), index=False)
            except EmptyDataError:
                logging.info("file with Teff={0:s} , \
                         vt = {1:s} km/s, logg = {2:s}, \
                        z = {3:s} does not exsit".format(teff, \
                        vt, logg, z))
                continue
    else:
        if grid_readme == None:
            raise ValueError("You must assign a readme of your grid!")
        rm = pd.read_csv(grid_readme, delimiter=",", header=0)
        pars = np.array([np.arange(r.start, r.end, r.step) for i,r in rm.iterrows()])
        for teff in tqdm.tqdm(pars[0], desc="Teff", leave=False):
            for logg in pars[1]:
                for z in pars[2]:
                    for vt in pars[3]:
                        if z>=0.0:
                            outname = gridpath+"{0:d}K/logg{1:.2f}/z{2:.2f}/out.fe.v8.1_{0:d}_" \
                                "+{1:.2f}_+{2:.2f}_{3:.2f}_*".format(int(teff), logg, z, vt)
                        else:
                            outname = gridpath+"{0:d}K/logg{1:.2f}/z{2:.2f}/out.fe.v8.1_{0:d}_" \
                                "+{1:.2f}_{2:.2f}_{3:.2f}_*".format(int(teff), logg, z, vt)
                        v1 = os.popen("grep -n -o 'KR' {0:s}".format(outname)).read()
                        v2 = os.popen("grep -n -o '1 D LG TAUNY' {0:s}".format(outname)).read()
                        if len(re.split("\n|:", v1))!=5 or len(re.split("\n|:", v2))!=5:
                            logger.info("Your output file: {0:s} from MULTI is incomplete.".format(outname))
                            continue

                        skiprows = int(re.split("\n|:", v1)[2]) + 1
                        nrows = int(re.split("\n|:", v2)[2]) - skiprows - 1
                        try:
                            store.append(gridpath+"{0:d}K/logg{1:.2f}/z{2:.2f}/vt{3:.2f}".format(int(teff), logg, z, vt),
                                read_outputfile(os.popen("ls {0:s}".format(outname)).read().rstrip(), skiprows=skiprows,
                                nrows=nrows, obs_linelist=obs_linelist, level_model=level_model), index=False)
                        except IndexError:
                            print(outname+" has some issue with RT calculation!")
                            #import pdb
                            #pdb.set_trace()
    store.close()#close hdf file
    #warnings.filters = original_warnings#recover python original warning

def slope_measure(x, y, yerr=None):

    def func_lw(x_lw, slope, offset):
        return slope*x_lw + offset

    # Determine type of sigma
    if yerr is not None:
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
        d = np.array([x, y, yerr])
    else:
        finite = np.isfinite(x) & np.isfinite(y)
        d = np.array([x, y])

    if np.sum(finite) == 0:
        raise ValueError("Your input array are all infinity and none values.")

    d = d[:, finite]
    xbar = np.mean(d[0])
    d[0] = d[0] - xbar
    if yerr is not None:
        popt, pcov = curve_fit(func_lw, d[0], d[1], sigma=d[2])
    else:
        popt, pcov = curve_fit(func_lw, d[0], d[1])

    return popt, pcov

def generate_ranges(stellar_type):
    #define range of stellar parameters for each categories
    Teff_ranges = {'M' : [2400.0, 4000.0],'K' : [4000.0,5200.0], 'G' : [5200.0, 6000.0], 'F' : [6000.0, 6850.0]}
    logg_ranges = {'dwarf': [4.0, 5.0], 'subgiant': [3.0,4.0], 'giant': [0.5, 3.0], 'supergiant': [0.0, 0.5]}
    feh_ranges = {'very_metal_poor': [-3.5, -2.0], 'metal_poor': [-2.0, -0.5], 'metal_rich': [-0.5, 0.5]}

    #generate stellar parameter ranges according to stellar_type
    if "whole_grid" in stellar_type:
        Teff_range = [4000, 6850]
        logg_range = [0.0, 5.0]
        feh_range = [-3.5, 0.5]
    elif all([s.replace("_", "").isalpha() for s in stellar_type.split("/")]):
        Teff_range = Teff_ranges[stellar_type.split('/')[0]]
        logg_range = logg_ranges[stellar_type.split('/')[1]]
        feh_range = feh_ranges[stellar_type.split('/')[2]]
    else:
        Teff_range = [float(stellar_type.split("/")[0].split("_")[0]), float(stellar_type.split("/")[0].split("_")[1])]
        logg_range = [float(stellar_type.split("/")[1].split("_")[0]), float(stellar_type.split("/")[1].split("_")[1])]
        feh_range = [float(stellar_type.split("/")[2].split("_")[0]), float(stellar_type.split("/")[2].split("_")[1])]

    return Teff_range, logg_range, feh_range

def check_on_the_edge(xl, funl, bounds):
    final_x = xl[0]
    final_fun = 1e5
    for i, xi in enumerate(xl):
        if ((xi[0] == bounds[0][0]) or (xi[0] == bounds[0][1]) or
            (xi[1] == bounds[1][0]) or (xi[1] == bounds[1][1]) or
            (xi[2] == bounds[2][0]) or (xi[2] == bounds[2][1])):
            pass
        else:
            if final_fun > funl[i]:
                final_x = xi
                final_fun = funl[i]
    return final_x, final_fun

def test_spectralib_complete(lib):
    hdf = pd.HDFStore(lib)
    for (path, subgroups, subkeys) in hdf.walk():
        #for subgroup in subgroups:
        #    print('GROUP: {}/{}'.format(path, subgroup))
        #    print(subgroup)
        for subkey in subkeys:
            key = '/'.join([path, subkey])
            print('KEY: {}'.format(key))
            print(hdf.get(key))

def docs_setup():
    """
    Set some environment variables and ignore some warnings for the docs.
    From https://github.com/exoplanet-dev/exoplanet/blob/main/src/exoplanet/utils.py
    """
    import logging
    import warnings

    import matplotlib.pyplot as plt

    # Remove when Theano is updated
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Remove when arviz is updated
    warnings.filterwarnings("ignore", category=UserWarning)

    logger = logging.getLogger("theano.gof.compilelock")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("theano.tensor.opt")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("exoplanet")
    logger.setLevel(logging.DEBUG)

    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rc('axes', linewidth=2)
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True
    plt.rcParams['xtick.direction']="in"
    plt.rcParams['ytick.direction']="in"
