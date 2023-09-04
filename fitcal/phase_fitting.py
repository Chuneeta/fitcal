from mwa_qa.read_calfits import CalFits
import numpy as np
from astropy.io import fits
from scipy import stats
from astropy import units as u
from astropy.constants import c
from scipy.optimize import minimize
from typing import NamedTuple
import pylab
import copy
pol_str = {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3}


class PhaseFitInfo(NamedTuple):
    """
        Contains info on the Phase Fit parameters
        """

    length: float
    intercept: float
    sigma_resid: float
    chi2dof: float
    quality: float
    stderr: float


class Phase_Fit(object):
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
        self.solutions = [
            self.gain_array[:, :, :, 0],
            self.gain_array[:, :, :, 1],
            self.gain_array[:, :, :, 2],
            self.gain_array[:, :, :, 3]
        ]
        self.convergence = cal.convergence[0]

    def wrap_angle(self, angle):
        """
        Wraps phase angle

        angle: NDarray of floats (angles/phases)
        """

        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def _phase_fit_line(self, per_antenna_solutions, fit_iono=None):
        """
        Fits the line to the phases for a single antenna and polarization

        per_antenna_solution: NDArray of shape (nfreqs)
    fit_iono: Boolean, if True, accounts for ionospheric shift in phase. Default is False

    Credit: snipplet taken from Sam McSweeney
        """

        # using normalized convergence results as weights
        weights = (self.convergence - np.nanmin(self.convergence)) / \
            (np.nanmax(self.convergence) - np.nanmin(self.convergence))
        # masking non-nan values
        mask = np.where(np.logical_and(np.isfinite(
            per_antenna_solutions), weights > 0))[0]

        masked_solution = per_antenna_solutions[mask]
        masked_freqs = self.frequencies[mask]
        masked_weights = weights[mask]

        # normalizing solution
        masked_solution /= np.abs(masked_solution)
        masked_solution *= masked_weights

        # sampling the frequencies
        dv = np.min(np.diff(masked_freqs)) * u.Hz
        v = masked_freqs * u.Hz
        bins = np.round((v/dv).decompose().value).astype(int)
        ctr_bin = (np.min(bins) + np.max(bins))//2
        shifted_bins = bins - ctr_bin

        dm = 0.01 * u.m
        dt = dm / c
        vmax = 0.5 / dt
        N = 2*int(np.round(vmax/dv))  # Nyquist theorem

        # adding paading
        shifted_bins[shifted_bins < 0] += N
        sol0 = np.zeros((N,)).astype(complex)
        sol0[shifted_bins] = masked_solution

        # Fourier Inverse
        isol0 = np.fft.ifft(sol0)
        t = -np.fft.fftfreq(len(sol0), d=dv.to(u.Hz).value) * u.s
        d = np.fft.fftshift(c*t)
        isol0 = np.fft.fftshift(isol0)

        imax = np.argmax(np.abs(isol0))
        dmax = d[imax]
        slope = (2*np.pi*u.rad*dmax/c).to(u.rad/u.Hz)

        if fit_iono:
            def model(v, m, c, alpha):
                return np.exp(1j*(m*ν + c + alpha/v**2))

            y_intercept = np.angle(
                np.mean(masked_solution/model(v.to(u.Hz).value, slope.value, 0, 0)))
            params = (slope.value, y_intercept, 0)
        else:
            def model(v, m, c):
                return np.exp(1j * (m*v + c))

            y_intercept = np.angle(
                np.mean(masked_solution/model(v.to(u.Hz).value, slope.value, 0)))
            params = (slope.value, y_intercept)

        def objective(params, v, data):
            constructed = model(v, *params)
            residuals = self.wrap_angle(np.angle(data) - np.angle(constructed))
            cost = np.sum(np.abs(residuals)**2)

            return cost

        res = minimize(objective, params, args=(
            v.to(u.Hz).value, masked_solution))
        params = res.x
        constructed = model(v.to(u.Hz).value, *params)
        residuals = np.angle(masked_solution) - np.angle(constructed)
        chi2dof = np.sum(np.abs(residuals)**2) / (len(residuals) - len(params))
        resid_std = residuals.std()
        resid_var = residuals.var(ddof=len(params))
        stderr = np.sqrt(np.diag(res.hess_inv * resid_var))
        mask = np.where(np.abs(residuals) < 2 * stderr[0])[0]
        solution = masked_solution[mask]
        v = v[mask]

        period = ((params[0] * u.rad / u.Hz) / (2 * np.pi * u.rad)).to(u.s)
        quality = len(mask) / len(self.frequencies)

        return PhaseFitInfo(
            length=(c * period).to(u.m).value,
            intercept=self.wrap_angle(params[1]),
            sigma_resid=resid_std,
            chi2dof=chi2dof,
            quality=quality,
            stderr=stderr[0]
        )

    def phase_fit_line(self, fit_iono=False):
        """
        Phase fit across all tiles and XX and YY polarizations
        """

        _sh = self.solutions[0].shape  # shape of one polarization
        fit_array = copy.deepcopy(np.angle(self.gain_array))
        # fit_array = np.zeros((_sh[0], _sh[1], _sh[2], 2))
        quality_metrics = {'XX': {'quality': [], 'stderr': []},
                           'YY': {'quality': [], 'stderr': []}}
        weights = (self.convergence - np.nanmin(self.convergence)) / \
            (np.nanmax(self.convergence) - np.nanmin(self.convergence))
        for ip, p in enumerate(['XX', 'YY']):
            quality_value = 0
            stderr_value = 0
            for t_id in range(_sh[1]):
                try:
                    for t in range(_sh[0]):
                        mask = np.where(np.logical_and(np.isfinite(
                            self.solutions[pol_str[p]][t, t_id, :]), weights > 0))[0]
                        phase_fit = self._phase_fit_line(
                            self.solutions[pol_str[p]][t, t_id, :])
                        gradient = (
                            2 * np.pi * u.rad * (phase_fit.length * u.m) / c).to(u.rad/u.Hz).value
                        model_freqs = np.linspace(np.nanmin(self.frequencies[mask]), np.nanmax(
                            self.frequencies[mask]), len(self.frequencies))
                        model = gradient * model_freqs + phase_fit.intercept
                        fit_array[t, t_id, :, pol_str[p]] = model
                        quality_value += phase_fit.quality
                        stderr_value += phase_fit.stderr
                    quality_metrics[p]['quality'].append(
                        quality_value / _sh[0])
                    quality_metrics[p]['stderr'].append(stderr_value / _sh[0])
                except ValueError:
                    continue
        return fit_array, quality_metrics

    def plot_phase_fit(self, ndarray, quality_metrics=None, figure_name=None):
        """
        Plots phase fits

        ndarray: NDArray with the phase fit values (time, ntiles, nfreqs, npol)
        """

        ncols = 16
        nrows = len(self.tile_ids) // ncols
    # plot xx polarizations
        fig, axs = pylab.subplots(nrows, ncols, figsize=(
            16, 9), sharex=True, sharey='row', squeeze=True)
        pylab.suptitle('Calibration solutions -- XX', size=7)
        for id, t_id in enumerate(self.tile_ids):
            ax = pylab.subplot(nrows, ncols, id + 1)
            ax.scatter(self.frequencies * 1e-6, self.wrap_angle(np.angle(
                self.solutions[0][0, t_id, :])), c='dodgerblue', alpha=0.8, s=5, marker='.')
            ax.plot(self.frequencies * 1e-6,
                    self.wrap_angle(ndarray[0, t_id, :, 0]), 'k-')
            ax.set_ylim(-np.pi, np.pi)
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)

        pylab.tight_layout()

        if figure_name is None:
            outfile = 'solutions_phasefit_XX.png'
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
            ax.scatter(self.frequencies * 1e-6, self.wrap_angle(np.angle(
                self.solutions[3][0, t_id, :])), c='dodgerblue', alpha=0.9, s=5, marker='.')
            ax.plot(self.frequencies * 1e-6,
                    self.wrap_angle(ndarray[0, t_id, :, 3]), 'k-')
            ax.set_ylim(-np.pi, np.pi)
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)

        pylab.tight_layout()

        if figure_name is None:
            outfile = 'solutions_phasefit_YY.png'
        else:
            if (figure_name.split('.')[-1] == '.png'):
                outfile = figure_name.replace('.png', '_YY.png')
            else:
                outfile = figure_name + '_YY.png'

        pylab.savefig(outfile, dpi=400)
        pylab.close()
