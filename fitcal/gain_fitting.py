from mwa_qa.read_calfits import CalFits
from fitcal import amplitude_fitting, phase_fitting
from astropy.io import fits
import numpy as np


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
        cal = CalFits(self.calfits)
        self.tile_ids = cal.antenna
        self.tile_names = cal.annames
        self.frequencies = cal.frequency_array
        self.gain_array = cal.gain_array

    def get_fitted_solutions(self, order, window, weights=None):
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
        """

        # smoothing the amplpitude
        famp = amplitude_fitting.Amplitude_Fit(self.calfits)
        smoothed_amp, amp_qa = famp.savgol_smoothing(order, window)
        print(smoothed_amp.shape)
        # fitting line to the phases
        fphs = phase_fitting.Phase_Fit(self.calfits)
        fitted_phase, phase_qa = fphs.phase_fit_line(self.calfits)
        print(fitted_phase.shape)

        self.fitted_solutions = smoothed_amp * \
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
