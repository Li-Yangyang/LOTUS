#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:42:20 2020

@author: yangyangli
"""

import numpy as np
import matplotlib.pylab as plt
#import matplotlib.ticker as tck
plt.rc('axes', linewidth=2)
plt.rcParams['xtick.minor.visible']=True
plt.rcParams['ytick.minor.visible']=True

def plot_optimized_equilibrium(star, opt_stellarpars, fit_pars,
                               REWs1, REWs2, chis1, chis2,
                               abunds1, abunds2, abunds1_err=None, abunds2_err=None):
    """
    Plot for abundances vs reduced EWs and abundances vs excitation potential given
    stellar parameters.

    Parameters
    ----------
    star : str
        Name of star
    opt_stellarpars : dict
        Contains Teff, logg, vt and their uncertainty
    fit_pars : list
        [dict(linear fitting of A vs REW), dict(linear fitting of A vs chi)]
    REWs1 : list or ndarray
        Reduced EWs for FeI
    REWs2 : list or ndarray
        Reduced EWs for FeII
    chis1 : list or ndarray
        Excitation potential for FeI
    chis2 : list or ndarray
        Excitation potential for FeI
    abunds1 : list of ndarry
        Abundances of FeI
    abunds2 : list of ndarry
        Abundances of FeI
    abunds1_err :list of ndarray, optional
        Error of derived abundances of FeI (not implemeted yet). The default is None.
    abunds2_err : list of ndarray, optional
        Error of derived abundances of FeII (not implemeted yet). The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        plot for equiibira

    """
    title = "%s\n $T_{eff}$ = %.2f $\pm$ %.2f K,\
    $log$g = %.2f $\pm$ %.2f (cm $\\cdot$ $s^{-2}$),\
    $\\left[Fe/H\\right]$ = %.2f $\pm$ %.2f,\
    $V_{mic}$ = %.2f $\pm$ %.2f km/s" %(star, opt_stellarpars["Teff"][0], opt_stellarpars["Teff"][1],
                                      opt_stellarpars["logg"][0], opt_stellarpars["logg"][1],
                                      opt_stellarpars["feh"][0], opt_stellarpars["feh"][1],
                                      opt_stellarpars["Vmic"][0], opt_stellarpars["Vmic"][1])

    from scipy.optimize import curve_fit

    def func_lw(x_lw, slope, offset):
        return slope*x_lw + offset

    if abunds1_err is not None:
        popt_Achi1, pcov_Achi1 = curve_fit(func_lw, chis1, abunds1, sigma=abunds1_err)
        popt_AREW1, pcov_AREW1 = curve_fit(func_lw, REWs1, abunds1, sigma=abunds1_err)
    else:
        popt_Achi1, pcov_Achi1 = curve_fit(func_lw, chis1, abunds1)
        popt_AREW1, pcov_AREW1 = curve_fit(func_lw, REWs1, abunds1)

    fig, ax = plt.subplots(2,1, figsize=(12,7))

    for a in ax:
        a.linewidth = 5
        a.tick_params(axis="x", which="major", size=5, width=2.5, labelsize=10, direction="in")
        a.tick_params(axis="y", which="major", size=5, width=2.5, labelsize=10, direction="in")
        a.tick_params(axis='x', which='minor', size=3, width=1.5, labelsize=5, direction="in")
        a.tick_params(axis='y', which='minor', size=3, width=1.5, labelsize=5, direction="in")
        #axe.xaxis.set_minor_locator(tck.AutoMinorLocator())
        #axe.yaxis.set_minor_locator(tck.AutoMinorLocator())

    xs = [[REWs1, REWs2], [chis1, chis2]]
    ys = [abunds1, abunds2]
    yerrs = [abunds1_err, abunds2_err]
    popts = [popt_AREW1, popt_Achi1]
    cs = ["k", "r"]
    ax[0].set_title(title)
    xlabels = ["log(EW/$\lambda$)", "$\chi$(ev)"]
    for i, a in enumerate(ax):
        if (yerrs[i] is None) and (yerrs[i] is None):
            a.scatter(xs[i][0], ys[0], s=40, facecolors='none', edgecolors=cs[0], linewidth=2)
            a.scatter(xs[i][1], ys[1], s=40, facecolors='none', edgecolors=cs[1], linewidth=2)
        else:
            a.errorbar(xs[i][0], ys[0], yerrs[0], color=cs[0], fmt="o", capsize=2, alpha=1, ms=5)
            a.errorbar(xs[i][1], ys[1], yerrs[1], color=cs[1], fmt="o", capsize=2, alpha=1, ms=5)
        a.set_xlabel(xlabels[i], fontsize=15)
        a.set_ylabel("log$\epsilon$(Fe)", fontsize=15)
        pred_xs = np.array([np.min(np.concatenate(xs[i], axis=0)), np.max(np.concatenate(xs[i], axis=0))])
        pred_a1s = popts[i][1] + popts[i][0] * pred_xs
        a.plot(pred_xs, pred_a1s, color=cs[0], linestyle="--",
               linewidth=2, label="FeI log$\epsilon$(Fe)vs %s: %.2e $\pm$ %.2e" % (xlabels[i], fit_pars[i][0], fit_pars[i][1]))
        a.legend(loc="upper right", frameon=False)
        a.set_xlim(pred_xs[0]-0.02, pred_xs[1]+0.02)
    plt.subplots_adjust()

    return fig

def plot_results_brute(result, grid, best_vals=True, varlabels=None,
                       output=None):
    """Visualize the result of the optimization results.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, dict
        Whether to show the best values from the grid search (default is True).
        if this is a bool, then the parameters in result will be used for ploting;
        if this a dictionary then values in this dictionary will be used for ploting.

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        plot for element-wise objective function values at proposed grid points
    """
    from matplotlib.patches import Ellipse
    npars = len(result['var_names'])
    _fig, axes = plt.subplots(npars, npars, figsize=(11,9))

    if not varlabels:
        varlabels = result['var_names']
    if best_vals and isinstance(best_vals, bool):
        best_vals = np.transpose([result["ScipyOptimizeResult"].x,
                                  result["ScipyOptimizeResult"].stderrs])
    if best_vals and isinstance(best_vals, dict):
        best_vals = np.array([[best_vals["Teff"][0], best_vals["Teff"][1]],
                              [best_vals["logg"][0], best_vals["logg"][1]],
                              [best_vals["Vmic"][0], best_vals["Vmic"][1]]])

    for i, par1 in enumerate(result['var_names']):
        for j, par2 in enumerate(result['var_names']):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(grid[i]),
                        np.minimum.reduce(grid[3], axis=red_axis),
                        'o', ms=3, color="k")
                ax.set_ylabel(r'$\chi^{2}$', fontsize=15)
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals.all():
                    ax.axvline(best_vals[i][0], ls='dashed', color='r')
                    ax.axvspan(best_vals[i][0] - best_vals[i][1],
                               best_vals[i][0] + best_vals[i][1],
                               alpha=0.5, color='red')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(grid[3], axis=red_axis),
                        np.unique(grid[i]), 'o', ms=3, color="k")
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i], fontsize=15)
                if i != npars-1:
                    ax.set_xticks([])
                elif i == npars-1:
                    ax.set_xlabel(r'$\chi^{2}$', fontsize=15)
                if best_vals.all():
                    ax.axhline(best_vals[i][0], ls='dashed', color='r')
                    ax.axhspan(best_vals[i][0] - best_vals[i][1],
                               best_vals[i][0] + best_vals[i][1],
                               alpha=0.5, color='red')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(np.unique(grid[i]),
                                   np.unique(grid[j]))
                lvls1 = np.linspace(grid[3].min(),
                                    np.median(grid[3])/2.0, 7, dtype='int')
                lvls2 = np.linspace(np.median(grid[3])/2.0,
                                    np.median(grid[3]), 3, dtype='int')
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                cf = ax.contourf(X.T, Y.T, np.minimum.reduce(grid[3], axis=red_axis),
                            lvls, levels=np.arange(0, 50, 2), cmap="RdBu_r", extend="max")
                #cf.cmap.set_over('k')
                cf.set_clim(0, 50)
                ax.set_yticks([])
                if best_vals.all():
                    ax.axvline(best_vals[i][0], ls='dashed', color='r')
                    ax.axhline(best_vals[j][0], ls='dashed', color='r')
                    ax.plot(best_vals[i][0], best_vals[j][0], 'rs', ms=3)
                    ellipse = Ellipse(xy=(best_vals[i][0], best_vals[j][0]),
                                      width=best_vals[i][1], height=best_vals[j][1],
                                      edgecolor='r', fc='None', lw=2, alpha=1.0)
                    ax.add_patch(ellipse)


                if j != npars-1:
                    ax.set_xticks([])
                elif j == npars-1:
                    ax.set_xlabel(varlabels[i],fontsize=15)
                if j - i >= 2:
                    axes[i, j].axis('off')
    _fig.colorbar(cf, ax=axes, pad=0.1)

    if output is not None:
        plt.savefig(output)

    return _fig

def plot_ewdiff_vs_stellar_parameters(df, line):
    plt.rc('axes', linewidth=2)
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True
    fig, ax = plt.subplots(4, 1, figsize=(12,10))

    wl, exp, ion = line.split("_")
    for a in ax:
        a.linewidth = 5
        a.tick_params(axis="x", which="major", size=5, width=2.5, labelsize=10, direction="in")
        a.tick_params(axis="y", which="major", size=5, width=2.5, labelsize=10, direction="in")
        a.tick_params(axis='x', which='minor', size=3, width=1.5, labelsize=5, direction="in")
        a.tick_params(axis='y', which='minor', size=3, width=1.5, labelsize=5, direction="in")

    xlabels = ['$T_{eff}(K)$', '$log\mathcal{g}(cm\cdot s^{-2})$', '$[Fe/H]$', '$v_{mic}(km\cdot s^{-1})$']
    for i, p in enumerate(['Teff', 'logg', 'feh', 'vt']):
        ax[i].set_xlabel(xlabels[i], fontsize=15)
        groupby_df = df.groupby(p)
        x = list(groupby_df.indices.keys())
        y_lte, yerr_lte = groupby_df.mean()['delta_lte'].values, groupby_df.std()['delta_lte'].values
        y_nlte, yerr_nlte = groupby_df.mean()['delta_nlte'].values, groupby_df.std()['delta_nlte'].values
        ax[i].errorbar(x, y_lte, yerr = yerr_lte,
                         color="black", fmt="o", capsize=2, alpha=1, ms=3, label="LTE")
        ax[i].errorbar(x, y_nlte, yerr = yerr_nlte,
                         color="red", fmt="o", capsize=2, alpha=1, ms=3, label="Non-LTE")
        ax[i].axhline(0, ls="--", ms=3, c="grey")
        if i == 0:
            ax[i].set_title(r"$\lambda={0:s}(\AA)$, ExPot={1:s}ev, Ion={2:s}".format(wl,exp,ion), fontsize=20)
            ax[i].legend(frameon=False)

    fig.text(0.065, 0.5, r"$\overline{EW_{theo} - EW_{inter}}(m\AA)$", va='center', rotation='vertical', fontsize=15)
    plt.subplots_adjust(hspace=0.4)
    return fig
