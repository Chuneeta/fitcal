from mwa_qa.read_calfits import CalFits
from fitcal import amplitude_fitting, phase_fitting
from astropy.io import fits
import numpy as np
import pylab


class HyperSolnFits(object):

    def __init__(self, calfits):
        """
        Object that takes in .fits containing the calibration solutions
        file readable by astropy
        and initializes them as global varaibles

        Parameters
        calfits : .fits file containing the calibration solutions
        """

        self.calfits = calfits
        self.cal = CalFits(self.calfits)
        self.tile_ids = self.cal.antenna
        self.tile_names = self.cal.annames
        self.frequencies = self.cal.frequency_array
        self.gain_array = self.cal.gain_array

    def create_ampqa_dict(self, amp_qa):
        for p in ['XX', 'YY']:
            self.qa['AIC_AMP{}'.format(p)] = amp_qa[p]['aic']
            self.qa['BIC_AMP{}'.format(p)] = amp_qa[p]['bic']
            self.qa['MSE_AMP{}'.format(p)] = amp_qa[p]['mse']
            self.qa['CHISQ_AMP{}'.format(p)] = amp_qa[p]['chisq']

    def create_phsqa_dict(self, phase_qa):
        for p in ['XX', 'YY']:
            self.qa['STDERR_PHS{}'.format(p)] = np.nanmean(
                phase_qa[p]['stderr'])
            self.qa['QUALITY_PHS{}'.format(p)] = np.nanmean(
                phase_qa[p]['quality'])

    def get_fitted_solutions(self, order=3, window=99, weights=None, fit_amplitude=False, fit_phase=False):
        """s
        Get the fitted amplitude and phase of the gain solutions

        Parameters
        order  : integer
                                Order used to fit the polynomial; can range for 0 to any finite number e.g (0, 4, 7)
        window : integer
                Size of the window, number to points for Savitzky-Golay Filter
        weights: ndarray
                1-dimensional consisting of valuses as a meausre of importance or priority to 
                                the data values to the fitted. Shoulde be of the same length as data_y. If None,
                                no weights would be applied.
        fit_amplitude : boolean
                If True, will fit a polynomial of desired order to the amplitude of the gain solutions.
                If set to False, raw amplitude solutions will be used. Default is True

        fit_phase :  boolean
                If True, a linear fit will be perfomrmed on the phases. 
                If set to False, raw phases will be used. Default is True.
        """

        # dictionary to save the quality metrics of the fit
        self.qa = {}
        # smoothing the amplpitude
        if ((fit_amplitude is True) and (fit_phase is False)):
            print('Smoothing amplitude of calibration solutions ...')
            famp = amplitude_fitting.Amplitude_Fit(self.calfits)
            fitted_amp, amp_qa = famp.savgol_smoothing(order, window)
            fitted_phase = self.cal.phases
            self.create_ampqa_dict(amp_qa)

        # fitting line to the phases
        elif ((fit_amplitude is False) and (fit_phase is True)):
            print('Fitting phase of calibration solutions ...')
            fphs = phase_fitting.Phase_Fit(self.calfits)
            fitted_phase, phase_qa = fphs.phase_fit_line(self.calfits)
            fitted_amp = self.cal.amplitudes
            self.create_phsqa_dict(phase_qa)

        # fitting both amplitude and phaase
        elif ((fit_amplitude is True) and (fit_phase is True)):
            print('Smoothing amplitude of calibration solutions ...')
            famp = amplitude_fitting.Amplitude_Fit(self.calfits)
            fitted_amp, amp_qa = famp.savgol_smoothing(order, window)
            print('Fitting phase of calibration solutions ...')
            fphs = phase_fitting.Phase_Fit(self.calfits)
            fitted_phase, phase_qa = fphs.phase_fit_line(self.calfits)
            self.create_ampqa_dict(amp_qa)
            self.create_phsqa_dict(phase_qa)

        else:
            fitted_amp = self.cal.amplitudes
            fitted_phase = self.cal.phases

        self.fitted_solutions = fitted_amp * \
            (np.cos(fitted_phase) + np.sin(fitted_phase) * 1j)

    def write_to(self, outfile=None, overwrite=True):
        """
        Writing fitted solutions to fits file

        Parameters
        outfile : str
            Name of the output file name. Default is the name of the input fitsfile with 
            a '_fitted' extension.
        """

        if outfile is None:
            outfile = self.calfits.replace('.fits', '_fitted.fits')
        try:
            with fits.open(self.calfits) as hdus:
                hdus['SOLUTIONS'].data[:, :, :, ::2] = np.real(
                    self.fitted_solutions)
                hdus['SOLUTIONS'].data[:, :, :, 1::2] = np.imag(
                    self.fitted_solutions)
                hdus.writeto(outfile, overwrite=overwrite)
        except AttributeError:
            print('Fit Object has no fitting attribute. Fitting of the data is required. No file is being written.')

        # adding qulaity metrics to the header
        with fits.open(outfile, 'update') as hdu:
            header = hdu['SOLUTIONS'].header
            for key in self.qa.keys():
                header[key] = self.qa[key]

    def plot_solutions(self, figure_name=None):
        """
        Plotting fitted values obtained from Polynomial Fitting or Smoothing Filter

        Parameters
        figure_name : str ; optional
                Name of ouptut figure
        """

        ncols = 16
        nrows = len(self.tile_ids) // ncols
        # plot xx polarizations
        fig, axs = pylab.subplots(nrows, ncols, figsize=(
            20, 12), sharex=True, sharey='row', squeeze=True)
        pylab.suptitle('Amplitude', size=7)
        for id, t_id in enumerate(self.tile_ids):
            ax = pylab.subplot(nrows, ncols, id + 1)
            ax.scatter(self.frequencies * 1e-6,
                       np.abs(self.fitted_solutions[0, t_id, :, 0]), c='blue', alpha=0.9, s=5, marker='.')
            ax.scatter(self.frequencies * 1e-6,
                       np.abs(self.fitted_solutions[0, t_id, :, 3]), c='red', alpha=0.9, s=5, marker='.')
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)
            ax.set_ylim(0, 2)

        pylab.tight_layout()

        if figure_name is None:
            outfile = self.calfits.replace('.fits', '_fitted_amplitude.png')
        else:
            if (figure_name.split('.')[-1] == '.png'):
                outfile = figure_name.replace('.png', '_amplitude.png')

        pylab.savefig(outfile, dpi=400)
        pylab.close()

        # plot yy polarizations
        fig, axs = pylab.subplots(nrows, ncols, figsize=(
            16, 9), sharex=True, sharey='row', squeeze=True)
        pylab.suptitle('Phase', size=7)
        for id, t_id in enumerate(self.tile_ids):
            ax = pylab.subplot(nrows, ncols, id + 1)
            ax.scatter(self.frequencies * 1e-6,
                       np.angle(self.fitted_solutions[0, t_id, :, 0]), c='blue', alpha=0.9, s=5, marker='.')
            ax.scatter(self.frequencies * 1e-6,
                       np.angle(self.fitted_solutions[0, t_id, :, 3]), c='red', alpha=0.9, s=5, marker='.')
            ax.tick_params(direction='in', labelsize=6)
            ax.set_title('{}|{}'.format(self.tile_names[id], t_id), size=6)
            ax.set_ylim(-np.pi, np.pi)

        pylab.tight_layout()

        if figure_name is None:
            outfile = self.calfits.replace('.fits', '_fitted_phase.png')
        else:
            if (figure_name.split('.')[-1] == '.png'):
                outfile = figure_name.replace('.png', '_phase.png')

        pylab.savefig(outfile, dpi=400)
        pylab.close()
