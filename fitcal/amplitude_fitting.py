from astropy.io import fits
from mwa_qa.read_calfits import CalFits
from numpy.polynomial import Polynomial
from scipy.signal import savgol_filter
import numpy as np
import copy
import pylab

# Number of coarse bands
Nband = 24
polstrs = ['XX', 'XY', 'YX', 'YY']


class Amplitude_Fit(object):
    def __init__(self, calfits):
        """
        Object that takes in .fits containing the calibration solutions
        file readable by astropy
        and initializes them as global varaibles

        Parameters
        calfits : .fits file containing the calibration solutions
        """
        self.calfits = calfits
        cal = CalFits(self.calfits)
        self.tile_ids = cal.antenna
        self.tile_names = cal.annames
        self.frequencies = cal.frequency_array
        self.gain_array = cal.gain_array
        self.ntime = cal.Ntime
        self.convergence = cal.convergence
        self.npols = self.gain_array.shape[-1]
        self.nfreq = len(self.frequencies)

    def quality_metrics(self, observed, expected):
        """
        Metrics for the goodness of fit

        Parameters
        observed : ndarray
                ND Array containing the observed data points
        expected : ndarray
                ND Array containing the expected data points (obstained from fitting, smoothing algorithm)

        Returns dictionary with
        mse : float
                Mean Square Error
        std : float
                Stanadard deviation of the residuals
        chisq : float
                chisq representing the goodness of fit
        """

        residuals = observed - expected
        mse = np.sum(np.abs(residuals)**2) / len(residuals)
        std = residuals.std()
        chisq = np.sum(np.abs(residuals)**2) / expected

        return {'mse': mse, 'std': std, 'chisq': chisq}

    def _polynomial_filter(self, data_x, data_y, order, weights):
        """
        Fitting a polynomial of desried order

        Parameters
        data_x : ndarray
                         1-d array consiting of values for the corresponsing data to be fitted
        data_y : ndarrray
                         1-d array consting of values to be fitted. Shoudl be of same length as data_x
        order  : integer
                         Order used to fit the polynomial; can range for 0 to any finite number e.g (0, 4, 7)		
        weights: ndarray
                         1-dimensional consisting of valuses as a meausre of importance or priority to 
                         the data values to the fitted. Shoulde be of the same length as data_y. If None,
                         no weights would be applied.

        Returns
        polynomial instance : instance/function of the fit that can be used to predict the y values for any 
                         given x value
        """
        fit_data = Polynomial.fit(data_x, data_y, deg=order, w=weights)
        return fit_data

    def _savgol_filter(self, data, order, window):
        """
        Using Savitzky-Golay Filter (piecewise polynomial fitting)

        Parameters
        data : ndarray  
            1-dimensional NDAarray with values to fit for
        order : integer
                Order used to fit the polynomial; can range for 0 to any finite number e.g (0, 4, 7)
        window :integer
                size of the window, number to points
        """

        return savgol_filter(data, window, order)

    def polynomial_fit(self, order, weigths=None):
        """
        Applying Polynomial Fitting to the hyperderive solutions

        Parameters
        order : integer
                Order used to fit the polynomial; can range for 0 to any finite number e.g (0, 4, 7)
        weights: ndarray
                         1-dimensional consisting of valuses as a meausre of importance or priority to 
                         the data values to the fitted. Shoulde be of the same length as data_y. If None,

        Returns
        poly_array : ndarray
                ND Array with fitted values
        qaulity_metrics : ndarray
                Dictionary of quality metrics drawn from the fitting
        """

        poly_array = copy.deepcopy(self.gain_array)

        for t in range(self.ntime):
            # filter any channels where result is NaN
            good_chs = np.where(~np.isnan(self.convergence[0]))[0]
            # Remove flagged tiles which are nan at first unflagged frequency and pol
            good_tiles = np.where(
                ~np.isnan(self.gain_array[t, :, good_chs[0], 0]))[0]
            for ant in good_tiles:
                for pol in range(self.npols):
                    fit_data = self._polynomial_filter(
                        good_chs, np.abs(self.gain_array[t, ant, good_chs, pol]), order=order, weights=weigths)
                    poly_array[t, ant, good_chs, pol] = fit_data(good_chs)

        quality_metrics = self.quality_metrics(self.gain_array, poly_array)

        return poly_array, quality_metrics

    def savgol_smoothing(self, order, window):
        """
        Applying Savitzky-Golay Filter to the hyperderive solutions

        Parameters
        order : integer
                Order used to fit the polynomial; can range for 0 to any finite number e.g (0, 4, 7)
        window :integer
                size of the window, number to points

        Returns
        poly_array : ndarray
                ND Array with fitted values
        qaulity_metrics : ndarray
                Dictionary of quality metrics drawn from the fitting
        """

        smoothed_array = copy.deepcopy(self.gain_array)
        for t in range(self.ntime):
            # filter any channels where result is NaN
            good_chs = np.where(~np.isnan(self.convergence[0]))[0]
            # Remove flagged tiles which are nan at first unflagged frequency and pol
            good_tiles = np.where(
                ~np.isnan(self.gain_array[t, :, good_chs[0], 0]))[0]
            for ant in good_tiles:
                for pol in range(self.npols):
                    smooth_data = self._savgol_filter(np.abs(self.gain_array[t, ant, good_chs, pol]), order=order,
                                                      window=window)
                    smoothed_array[t, ant, good_chs, pol] = smooth_data

        quality_metrics = self.quality_metrics(self.gain_array, smoothed_array)

        return smoothed_array, quality_metrics

    def plot_polynomial_fit(self, fitted_array, ylim=[0, 2], figure_name=None):
        """
        Plotting fitted values obtained from Polynomial Fitting or Smoothing Filter

        Parameters
        fitted_array : ndarray 
                NDArray with the fitted values (time, ntiles, nfreqs, npol)
        ylim : list
                List of length 2 with the values for the extent on the yaxis
        figure_name : str ; optional
                Name of ouptut figure
        """

        ncols = 16
        nrows = len(self.tile_ids) // ncols
        # plot xx polarizations
        fig, axs = pylab.subplots(nrows, ncols, figsize=(
            16, 9), sharex=True, sharey='row', squeeze=True)
        pylab.suptitle('Calibration solutions -- XX', size=7)
        for id, t_id in enumerate(self.tile_ids):
            ax = pylab.subplot(nrows, ncols, id + 1)
            ax.scatter(self.frequencies * 1e-6, np.abs(
                self.gain_array[0, t_id, :, 0]), c='dodgerblue', alpha=0.9, s=5, marker='.')
            ax.plot(self.frequencies * 1e-6,
                    fitted_array[0, t_id, :, 0], 'k-', alpha=0.8)
            ax.set_ylim(ylim[0], ylim[1])
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)

        pylab.tight_layout()

        if figure_name is None:
            outfile = 'solutions_ampfit_XX.png'
        else:
            if (figure_name.split('.')[-1] == '.png'):
                outfile = figure_name.replace('.png', '_XX.png')
            else:
                outfile = figure_name + '_XX.png'

        pylab.savefig(outfile, dpi=400)
        pylab.close()

        # plot yy polarizations
        fig, axs = pylab.subplots(nrows, ncols, figsize=(
            16, 9), sharex=True, sharey='row', squeeze=True)
        pylab.suptitle('Calibration solutions -- YY', size=7)
        for id, t_id in enumerate(self.tile_ids):
            ax = pylab.subplot(nrows, ncols, id + 1)
            ax.scatter(self.frequencies * 1e-6, np.abs(
                self.gain_array[0, t_id, :, 3]), c='dodgerblue', alpha=0.9, s=5, marker='.')
            ax.plot(self.frequencies * 1e-6,
                    fitted_array[0, t_id, :, 3], 'k-', alpha=0.8)
            ax.set_ylim(ylim[0], ylim[1])
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)

        pylab.tight_layout()

        if figure_name is None:
            outfile = 'solutions_ampfit_YY.png'
        else:
            if (figure_name.split('.')[-1] == '.png'):
                outfile = figure_name.replace('.png', '_YY.png')
            else:
                outfile = figure_name + '_YY.png'

        pylab.savefig(outfile, dpi=400)
        pylab.close()
