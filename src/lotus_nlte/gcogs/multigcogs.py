#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:27:46 2021

@author: yangyangli
"""
import os, glob

import numpy as np
import pandas as pd
import h5py
import joblib
import tarfile

from astropy.stats.info_theory import bayesian_info_criterion_lsq

from .gcog import SingleGCOG
from .utils import *

#from ..interpolation import *
from ..config import *
from ..utils import generate_ranges

class MultiGCOG:
    """
    Base class for General Curve of Growth (GCOG) of multiple lines

    This intends for assemble GCOG of multiple lines.

    Parameters
    ----------
    star: str
        The name of target star
    stellar_type: str:
        The stellar type of your star, like:
            {spectral type, e.g. F, G, K}/{giant or subgiant or dwarf}/{metal_rich or metal_poor or very_metal_poor} or
        the estimation of your atmospheric parameters in such form:
            {{T_low}_{T_high}/{logg_low}_{logg_high}/{feh_low}_{feh_high}}
    obs_path: str
        Path of the observation ew list of the target star
    cal: str
        Types of derivation, e.g. "lte" or "nlte"
    exp_cutoff: int or float
        Cutoff of excitation potential during the derivation, in the unit of ev
    ewlibpath: str
        The path for the libary of EW, it must be a h5 file
    interpolation: bool, default: False
        True: use interpolated GCOG
        False: get GCOG from EW library
        
    """

    __slots__ = ['star', 'stellar_type', 'exp_cutoff', "obs_wavelength", "obs_ep",
                 "obs_ele", "obs_ew", "cal", "models", "interp_method",
                 "ewlibpath", "working_dir", "interpolation"]

    def __init__(self, star, stellar_type, obs_path, cal="nlte", exp_cutoff=0,
                 ewlibpath=None, interpolation=False):
        self.star = star
        self.stellar_type = stellar_type
        self.exp_cutoff = exp_cutoff
        #read in obs linelist
        self._read_from_obs_linelist(obs_path)

        self.ewlibpath = ewlibpath

        self.cal = cal
        self.interpolation = interpolation


        if not self.interpolation:
            print("All optimizations are based on exist interpolated models and you don't need to interpolate them!")

            if self.cal == "lte":
                self.interptar_path = GCOG_LTE_LIB
            else:
                self.interptar_path = GCOG_NLTE_LIB
            interptar = tarfile.open(self.interptar_path)
            self.working_dir = interptar.getnames()[0] + "/" + self.stellar_type + "/"
        else:
            if not self.ewlibpath:
                raise ValueError("Please assign your EW library!")
            else:
                if self.cal == "lte":
                    self.working_dir = GCOG_LTE_DIR + self.stellar_type + "/"
                else:
                    self.working_dir = GCOG_NLTE_DIR + self.stellar_type + "/"

                self._keys, self._ini_cents = get_keys_and_atmos_pars(self.ewlibpath, self.stellar_type)

        self.models = []
        self.interp_method = "*"

    def pipelines(self):
        """
        Method that can select lines with accurate and precise abundance prediction

        Returns
        -------
        None.

        """
        #generate gcog or select gcog for each lines

        if not os.path.exists(self.working_dir) and self.interpolation :
            os.makedirs(self.working_dir)
            self._select_observed_gcogs()


            #Since the algorithm of multivariate polynomial regression hasn't been sped up
            #in sklearn toolkit, we have to calculte the model and dump them into files first
            #for later persistent prediction
        else:
            if self.interp_method == "[2-5]":
                interptar = tarfile.open(self.interptar_path)
                difftar = tarfile.open(self.difftar_path)
            exist_mask = []
            for i in range(len(self.obs_wavelength)):
                fname = find_closest_model(self.obs_wavelength[i], self.obs_ep[i],
                                           self.obs_ele[i], self.working_dir,
                                           self.interp_method, self.interpolation,
                                           interptar)
                if len(fname) > 0:
                    m = self._load_model([interptar, fname[0]])
                    line = fname[0].split("/")[4].split(".sav")[0][:-2]
                    if self._if_correct_interp(line, m, difftar):
                        self.models.append(m)
                        if self.interp_method == "SKIGP":
                            self.likelihoods.append(m.likelihood)
                        exist_mask.append(True)
                        print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} has already existed and passes test of interpolation".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                     self.obs_ele[i]))
                    else:
                        exist_mask.append(False)
                        print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} exists but doesn't pass test of interpolation".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                     self.obs_ele[i]))
                else:
                    try:
                        exist_mask.append(self._select_one_observed_gcogs(i))
                    except ValueError:
                        exist_mask.append(False)
                    except np.linalg.LinAlgError:
                        print("RBF model for line {0:.2f}A with ep={1:.2f}ev of element {2:s} is not working".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                     self.obs_ele[i]))
                        exist_mask.append(False)
                    except AttributeError:
                        print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} doesn't have enough points for interpolation".format(self.obs_wavelength[i],
                                                                                                 self.obs_ep[i],
                                                                                                 self.obs_ele[i]))
                        exist_mask.append(False)


            self._update_obslinelist(exist_mask)
            #self.models = np.array(self.models)
        if self.interp_method == "SKIGP":
            self.models = gpytorch.models.IndependentModelList(*self.models)
            self.likelihoods = gpytorch.likelihoods.LikelihoodList(*self.likelihoods)

    def _load_model(self, handle):
        """
        Interface for load the model, provide for subclass

        Parameters
        ----------
        handle : optional
            path or index or tarfile handle of interpolated model.
        """
        pass

    def _read_from_obs_linelist(self, obs_path):
        """
        Read observation EW line list

        Parameters
        ----------
        obs_path : str
            path of EW line list.
        """
        linelist = pd.read_csv(obs_path)
        idx_cutoff = ((linelist['obs_ep'] >= self.exp_cutoff) & (linelist['element'] == 'FeI')) | (linelist['element'] == 'FeII')
        self.obs_wavelength = np.array(linelist['obs_wavelength'][idx_cutoff])
        self.obs_ep = np.array(linelist['obs_ep'][idx_cutoff])
        self.obs_ele = np.array(linelist['element'][idx_cutoff])
        self.obs_ew = np.array(linelist['obs_ew'][idx_cutoff])


    def _update_obslinelist(self, mask):
        """
        Remove lines that don't have enough lines to interpolate or can't pass
        EW precision test

        Parameters
        ----------
        mask : list of True and False
            Indicate which lines are not available
            for later opimization

        """
        self.obs_wavelength = np.ma.masked_array(self.obs_wavelength, mask=np.invert(mask)).compressed()
        self.obs_ep = np.ma.masked_array(self.obs_ep, mask=np.invert(mask)).compressed()
        self.obs_ele = np.ma.masked_array(self.obs_ele, mask=np.invert(mask)).compressed()
        self.obs_ew = np.ma.masked_array(self.obs_ew, mask=np.invert(mask)).compressed()

        assert np.shape(self.obs_ele)[0] == \
            np.shape(self.obs_ep)[0] == np.shape(self.obs_ew)[0] == \
                np.shape(self.obs_wavelength)[0]

    def _select_one_observed_gcogs(self, i):
        """
        Assemble single GCOG

        Parameters
        ----------
        i : int
            Index of line.

        Returns
        -------
        bool
            mask value for the line

        """
        #TODO:parallelism this funciton?
        sg = SingleGCOG(self.obs_wavelength[i],self.obs_ep[i], self.obs_ele[i],
                        self.stellar_type, self.ewlibpath, self.cal, self._keys, self._ini_cents)
        s = sg.assemble_hyper_surface()

        if isinstance(s, np.ndarray):
            m = self._generate_model(s, self.obs_wavelength[i], self.obs_ep[i], self.obs_ele[i])
            #print("we have models!")
            #print infos change to logger
            line = format(self.obs_wavelength[i], ".2f") \
                +"_"+ format(self.obs_ep[i], ".2f") +"_" + self.obs_ele[i]
            if self._if_correct_interp(line, m):
                self.models.append(m)
                if self.interp_method == "SKIGP":
                    self.likelihoods.append(m.likelihood)
                print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} has been assembled successfully and passes test of interpolation ".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                     self.obs_ele[i]))
                del sg, s, m
                return True
            else:
                print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} has been assembled successfully but doesn't pass test of interpolation ".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                   self.obs_ele[i]))
                del sg, s, m
                return False
        else:
            print("Hypersurface of line {0:.2f}A with ep={1:.2f}ev of element {2:s} doesn't have enough points for interpolation".format(self.obs_wavelength[i],
                                                                                     self.obs_ep[i],
                                                                                     self.obs_ele[i]))
            del sg, s
            return False

    def _select_observed_gcogs(self):
        """
        Assemble GCOG for all lines

        """
        # if select high SNR obs lines?
        exist_mask = []
        for i, (wl, ep, ele) in enumerate(zip(self.obs_wavelength, self.obs_ep, self.obs_ele)):
            try:
                exist_mask.append(self._select_one_observed_gcogs(i))
            except ValueError:
                exist_mask.append(False)
            except np.linalg.LinAlgError:
                print("RBF model for line {0:.2f}A with ep={1:.2f}ev of element {2:s} is not working".format(wl,
                                                                                     ep, ele))
                exist_mask.append(False)


        self._update_obslinelist(exist_mask)

    def _if_correst_interp(self, line, m):
        """
        Judge if the interpolator accurate and precise enough to predict abundance
        of this star

        Parameters
        ----------
        line : str
            format:"wavelengh(2 decimal)_exp(2 decimal)_element(FeI or FeII)".
        m : object
            Type depends on the method of interpolation
        """
        #Interface for judge if the interpolation is accurate and precise, provide for subclass
        pass

    def remove_outliers(self, abunds, **kwargs):
        """
        Remove outliers according to the derived abundance

        Parameters
        ----------
        abunds : ndarray
            Derived abdunaces from GCOG

        """
        from astropy.stats import sigma_clip, biweight_scale

        if not "stdfunc" in kwargs:
            from astropy.stats import biweight_scale
            stdfunc = biweight_scale
        else:
            stdfunc = "std"

        idx_fei = np.where(np.array(self.obs_ele) =="FeI")
        idx_feii = np.where(np.array(self.obs_ele) =="FeII")

        #sigma clip
        clipped_bound1 = sigma_clip(abunds[idx_fei], stdfunc=stdfunc, return_bounds=True , **kwargs)[1:]
        idx_clipped1 = np.where(((abunds[idx_fei]>=clipped_bound1[0]) & (abunds[idx_fei]<=clipped_bound1[1])))[0]

        clipped_bound2 = sigma_clip(abunds[idx_feii], stdfunc=stdfunc, return_bounds=True, **kwargs)[1:]
        idx_clipped2 = np.where(((abunds[idx_feii]>=clipped_bound2[0]) & (abunds[idx_feii]<=clipped_bound2[1])))[0]

        #stack two clipped index
        idx_clipped = np.append(idx_fei[0][idx_clipped1], idx_feii[0][idx_clipped2])

        self.models = [self.models[x] for x in idx_clipped]
        self.obs_wavelength = self.obs_wavelength[idx_clipped]
        self.obs_ew = self.obs_ew[idx_clipped]
        self.obs_ep = self.obs_ep[idx_clipped]
        self.obs_ele = self.obs_ele[idx_clipped]

class PolyMultiGCOG(MultiGCOG):
    """
    Sub class for General Curve of Growth (GCOG) of multiple lines based on
    multivariate polynomial regresssion

    Parameters
    ----------
    ew_error: float or int
        Max of the deviation of EW allowed for the predicted EW from
        multivariate polynomial model
        if mean(pred)+std(pred) > ew_error or mean(prea)-std(pred) < -ew_error,
        the line doesn't pass the precision test

    """
    from ..interpolation.multipoly_interp import MultivariatePolynomialInterpolation
    def __init__(self, star, stellar_type, obs_path, exp_cutoff=0, ew_error=5,
                 ewlibpath=None,
                 ewdiffpath=None,
                 interpolation=False,
                 cal="nlte"):

        self.ewdiffpath = ewdiffpath
        self.ew_error = ew_error
        super(PolyMultiGCOG, self).__init__(star, stellar_type, obs_path, cal,
                         exp_cutoff, ewlibpath, interpolation)
        #create directory for save the polynomial test files
        self.interp_method = "[2-5]"
        if interpolation:
            if not os.path.exists(self.ewdiffpath+self.stellar_type):
                os.makedirs(self.ewdiffpath+self.stellar_type)
        else:
            self.difftar_path = EWDIFF_LIB

    def _load_model(self, handle):
        if isinstance(handle, str):
            return joblib.load(handle)
        if isinstance(handle[0], tarfile.TarFile) and isinstance(handle[1], str):
            return joblib.load(handle[0].extractfile(handle[1]))
        raise TypeError("Your model files path or type is not correct.")

    def _generate_model(self, s, wl, ep, ele):

        final_model = 0
        final_ewdiff_df = 0
        final_bic = np.inf
        final_n = 0
        for n in [2,3,4,5]:
            interpolator = MultivariatePolynomialInterpolation(s[:,[0,1,3,4]], s[:,2], degree=n)
            model = interpolator.fit()
            ewdiff_df = ewdiff(str(wl)+"_"+str(ep)+"_"+ele, self.stellar_type,
                           self.ewlibpath, oneline_model=model, cal=self.cal)
            idx_nonan = ~np.isnan(ewdiff_df["delta_"+self.cal])
            ssr = np.sum(np.array(ewdiff_df["delta_"+self.cal][idx_nonan]) ** 2.0)
            n_sample = len(idx_nonan)
            n_parameter = model[1].coef_.shape[0] + 1
            bic = bayesian_info_criterion_lsq(ssr, n_params=n_parameter, n_samples=n_sample)
            if bic < final_bic:
                final_bic = bic
                final_model = model
                final_ewdiff_df = ewdiff_df
                final_n = n
        print("Polynomial with degree={0:d} having BIC={1:.2f} is the optimal model.".format(final_n, final_bic))
        fmodel = self.working_dir + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele + "_" + str(final_n) + ".sav"
        joblib.dump(final_model, fmodel)
        fewdiff_df = self.ewdiffpath + self.stellar_type + "/" + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele  + ".csv"
        if not os.path.isfile(fewdiff_df):
            final_ewdiff_df.to_csv(fewdiff_df)
        del fewdiff_df, ewdiff_df
        return final_model

    def _if_correct_interp(self, line, m, difftar=None):
        if self.interpolation:
            ewdiff_file = self.ewdiffpath + self.stellar_type + "/" + line + ".csv"
            if os.path.isfile(ewdiff_file):
                ewdiff_df = pd.read_csv(ewdiff_file)
                if "delta_" + self.cal in ewdiff_df and "EW_" + self.cal in ewdiff_df:
                    mean, std = ewdiff_df["delta_"+self.cal].mean(), ewdiff_df["delta_"+self.cal].std()
                else:
                    ewdiff_df2 = ewdiff(line, self.stellar_type,
                               self.ewlibpath, oneline_model=m, cal=self.cal)
                    ewdiff_df["delta_"+self.cal] = ewdiff_df2["delta_"+self.cal]
                    ewdiff_df["EW_"+self.cal] = ewdiff_df2["EW_"+self.cal]
                    ewdiff_df.to_csv(ewdiff_file)
                    mean, std = ewdiff_df["delta_"+self.cal].mean(), ewdiff_df["delta_"+self.cal].std()
                    del ewdiff_df2
            else:
                ewdiff_df = ewdiff(line, self.stellar_type,
                           self.ewlibpath, oneline_model=m, cal=self.cal)
                ewdiff_df.to_csv(ewdiff_file)
                mean, std = ewdiff_df["delta_"+self.cal].mean(), ewdiff_df["delta_"+self.cal].std()

        else:
            ewdiff_file = difftar.getnames()[0] + "/" + self.stellar_type + "/" + line + ".csv"
            ewdiff_df = pd.read_csv(difftar.extractfile(ewdiff_file))
            mean, std = ewdiff_df["delta_"+self.cal].mean(), ewdiff_df["delta_"+self.cal].std()

        del ewdiff_df
        if (((mean+std)<self.ew_error) & ((mean-std)>-self.ew_error)):
            return True
        else:
            return False

class RBFMultiGCOG(MultiGCOG):
    """
    Sub class for General Curve of Growth (GCOG) of multiple lines based on
    nearest rbf regresssion

    Sub class parameters
    ----------
    met_error: float
        Max of the deviation of [Fe/H] allowed for the predicted [Fe/H] from
        nearest rbf model
        if mean(pred)+std(pred) > met_error or mean(prea)-std(pred) < -met_error,
        the line doesn't pass the precision test
    kernel: str
        Type of kernel for RBF regression, details in rbf package
        https://rbf.readthedocs.io/en/latest/installation.html
    k: int
        No of nearest points when interpolating
    """
    #from ..interpolation.rbf_interp import RBFRegressionInterpolation
    def __init__(self, star, stellar_type, obs_path, exp_cutoff=0, met_error=0.1,
                 ewlibpath="./LOTUS/EWLIB_largergrid2_v0.h5",
                 metdiffpath="./LOTUS/package_data/metdiff",
                 cal="nlte", kernel="mat32", k=50):

        self.metdiffpath = metdiffpath
        self.met_error = met_error
        super(RBFMultiGCOG, self).__init__(star, stellar_type, obs_path, cal,
                         exp_cutoff, ewlibpath)
        #create directory for save the polynomial test files
        self.kernel = kernel
        self.k = k
        self.interp_method = "RBF_" + self.kernel + "_" + str(self.k)
        if not os.path.exists(self.metdiffpath+self.stellar_type):
            os.makedirs(self.metdiffpath+self.stellar_type)

    def _load_model(self, handle):
        return joblib.load(handle)

    def _generate_model(self, s, wl, ep, ele):

        interpolator = RBFRegressionInterpolation(s[:,[0,1,3,4]], s[:,2], kernel=self.kernel, k=self.k)

        model = interpolator.fit()

        fmodel = self.working_dir + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele + "_" + self.interp_method +  ".sav"
        #test part
        idx = np.random.choice(range(np.shape(s)[0]), 1000)
        test_x = s[idx][:, [0,1,3,4]]
        test_y = s[idx][:, 2]
        met_diff = interpolator.test(test_x, test_y)
        #save the model
        joblib.dump(model, fmodel)
        fmetdiff_df = self.metdiffpath + self.stellar_type + "/" + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele  + "_" + self.interp_method + ".csv"
        if not os.path.isfile(fmetdiff_df):
            final_metdiff_df = pd.DataFrame(met_diff.T, columns=["delta_"+self.cal])
            final_metdiff_df["idx_"+self.cal] = idx.T
            final_metdiff_df.to_csv(fmetdiff_df)
        else:
            metdiff_df = pd.read_csv(fmetdiff_df)
            if ("delta_" + self.cal not in metdiff_df) or ("idx_" + self.cal not in metdiff_df):
               metdiff_df["delta_"+self.cal] = met_diff
               metdiff_df["idx_"+self.cal] = idx.T
               metdiff_df.to_csv(fmetdiff_df)

        return model

    def _if_correct_interp(self, line, m):
        metdiff_file = self.metdiffpath + self.stellar_type + "/" + line + "_" + self.interp_method + ".csv"
        if os.path.isfile(metdiff_file):
            metdiff_df = pd.read_csv(metdiff_file)
            if ("delta_" + self.cal in metdiff_df) and ("idx_" + self.cal in metdiff_df):
                mean, std = metdiff_df["delta_"+self.cal].mean(), metdiff_df["delta_"+self.cal].std()
            else:
                idx = np.random.choice(range(np.shape(m.y)[0]), 1000)
                test_x = m.y[idx]
                test_y = m.d[idx]
                met_diff = m(test_x) - test_y
                metdiff_df["delta_"+self.cal] = met_diff
                metdiff_df["idx_"+self.cal] = idx.T
                metdiff_df.to_csv(metdiff_file)
                mean, std = np.mean(met_diff), np.std(met_diff)
        else:
            idx = np.random.choice(range(np.shape(m.y)[0]), 1000)
            test_x = m.y[idx]
            test_y = m.d[idx]
            met_diff = m(test_x) - test_y
            final_metdiff_df = pd.DataFrame(met_diff.T, columns=["delta_"+self.cal])
            final_metdiff_df["idx_"+self.cal] = idx.T
            final_metdiff_df.to_csv(metdiff_file)
            mean, std = np.mean(met_diff), np.std(met_diff)

        if (((mean+std)<self.met_error) & ((mean-std)>-self.met_error)):
            return True
        else:
            return False



class SKIGpMultiGCOG(MultiGCOG):
    #from ..interpolation.gp_interp import GPInterpolation
    def __init__(self, star, stellar_type, obs_path, cal="nlte", exp_cutoff=0, met_error=0.3,
                 ewlibpath="./LOTUS/EWLIB_largergrid2_v0.h5",
                 metdiffpath="./LOTUS/package_data/metdiff"):

        self.met_error = met_error
        self.metdiffpath = metdiffpath

        super(SKIGpMultiGCOG, self).__init__(star, stellar_type, obs_path, cal, exp_cutoff, ewlibpath)
        self.interp_method = "SKIGP"
        self.likelihoods = []

        if not os.path.exists(self.metdiffpath+self.stellar_type):
            os.makedirs(self.metdiffpath+self.stellar_type)

    def _load_model(self, handle):
        #TODO:load tranning data waste time...if we can discard this part in future?
        
        sg = SingleGCOG(self.obs_wavelength[i],self.obs_ep[i], self.obs_ele[i],
                          self.stellar_type, self.ewlibpath, self.cal, self._keys, self._ini_cents)
        s = sg.assemble_hyper_surface()

        train_x = torch.from_numpy(s[:, [0,1,3,4]]).to(torch.float)
        train_y = torch.from_numpy(s[:, 2]).to(torch.float)

        state_dict = torch.load(fname)

        bounds = list(generate_ranges(self.stellar_type)[:2])
        bounds.append([0.5, 3.0])
        bounds.append([max(0, s[:,4].min()-10), s[:,4].max()+10])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood, bounds) # Create a new GP model

        model.load_state_dict(state_dict)

        return model

    def _generate_model(self, s, wl, ep, ele):
        interpolator = GPInterpolation(s[:, [0,1,3,4]], s[:,2], self.stellar_type)
        interpolator.train()
        met_diff = interpolator.test()
        fmodel = self.working_dir + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele + "_" + "SKIGP" + ".sav"
        torch.save(interpolator._model.state_dict(), fmodel)
        fmetdiff_df = self.metdiffpath + self.stellar_type + "/" + format(wl, ".2f") \
            +"_"+ format(ep, ".2f") +"_" + ele  + ".csv"

        if not os.path.isfile(fmetdiff_df):
            final_metdiff_df = pd.DataFrame(met_diff.T, columns=["delta_"+self.cal, "edelta_"+self.cal])
            final_metdiff_df.to_csv(fmetdiff_df)
        else:
            metdiff_df = pd.read_csv(fmetdiff_df)
            if "delta_" + self.cal not in metdiff_df:
               metdiff_df["delta_"+self.cal] = met_diff[:,0]
               metdiff_df["edelta_"+self.cal] = met_diff[:,1]
               metdiff_df.to_csv(fmetdiff_df)

        return interpolator._model

    def _if_correct_interp(self, line, m):
        metdiff_file = self.metdiffpath + self.stellar_type + "/" + line + ".csv"
        if os.path.isfile(metdiff_file):
            metdiff_df = pd.read_csv(metdiff_file)
            if "delta_" + self.cal in metdiff_df:
                mean, std = metdiff_df["delta_"+self.cal].mean(), metdiff_df["delta_"+self.cal].std()

            if (((mean+std)<self.met_error) & ((mean-std)>-self.met_error)):
                return True
            else:
                return False
