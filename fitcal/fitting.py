from astropy.io import fits
from numpy.polynomial import Polynomial
import numpy as np

# Number of coarse bands
Nband = 24
polstrs = ['XX', 'XY', 'YX', 'YY']

class Fit(object):
	def __init__(self, calfits):
		"""
		Object that takes in .fits containing the calibration solutions
		file readable by astropy
		and initializes them as global varaibles
		- calfits : .fits file containing the calibration solutions
		"""
		self.calfits = calfits
		with fits.open(self.calfits) as hdus:
			data = hdus['SOLUTIONS'].data
			self.gain_array = data[:, :, :, ::2] + data[:, :, :, 1::2] * 1j
			_sh = self.gain_array.shape
			self.convergence = hdus['RESULTS'].data
			self.Ntimes = _sh[0]
			self.Nants = _sh[1]
			self.Nfreq = _sh[2]
			self.Npols = _sh[3]
			self.Nband = Nband
			assert self.Nfreq % Nband == 0
			self.Nchan = self.Nfreq // self.Nband
			
	def unflagged_channels(self, edge_width, clip_width):
		# getting the unflagged channels
		channels = np.arange(self.Nchan)
		channels = np.delete(channels, self.Nchan // 2) # centre width
		channels = channels[edge_width + clip_width: -edge_width + clip_width]
		all_channels = []
		for bd in range(self.Nband):
			for ch in channels:
				all_channels.append(ch + (bd * self.Nchan))
		return all_channels

	def _polynomial_fit(self, data_x, data_y, order, weights):
		data_real = np.real(data_y)
		data_imag = np.imag(data_y)
		fit_real = Polynomial.fit(data_x, data_real, deg=order, w=weights)
		fit_imag = Polynomial.fit(data_x, data_imag, deg=order, w=weights)
		return fit_real, fit_imag

	def fit_data(self, order, weights=None, edge_width=2, clip_width=0):
		self.poly_array = np.zeros_like(self.gain_array, dtype=complex)
		unflagged_chs = self.unflagged_channels(edge_width, clip_width)
		cols = []
		times , antennas = [], []
		polarizations, complx = [], []
		coeffs = []
		for t in range(self.Ntimes):
			# filter any channels where result is NaN
			good_chs = [ch for ch in unflagged_chs if not np.isnan(self.convergence[t, ch])]
			# Remove flagged tiles which are nan at first unflagged frequency and pol
			good_tiles = np.argwhere(~np.isnan(self.gain_array[t, :, good_chs[0], 0]))
			for ant in good_tiles:
				for pol in range(self.Npols):
					fit_real, fit_imag = self._polynomial_fit(
						good_chs, self.gain_array[t, ant, good_chs, pol], order=order, weights=weights)
					self.poly_array[t, ant, :, pol] = fit_real(np.arange(self.Nfreq)) + fit_imag(np.arange(self.Nfreq)) * 1j
					# writing to fits colums
					for i, cp in enumerate(['REAL', 'IMAG']):
						times.append(t)
						antennas.append(ant[0])
						polarizations.append(polstrs[pol])
						complx.append(cp)
						coeffs.append(fit_real.coef.tolist()) if i % 2 == 0 else coeffs.append(fit_imag.coef.tolist())

		coeffs = np.array(coeffs)
		_sh = coeffs.shape
		cols.append(fits.Column(name = 'Timestamp', format='I', array=np.array(times)))
		cols.append(fits.Column(name = 'Antenna', format='I', array=np.array(antennas)))
		cols.append(fits.Column(name = 'Pol', format='2A', array=np.array(polarizations)))
		cols.append(fits.Column(name = 'Complex', format='4A', array=np.array(complx)))
		for cf in range(_sh[1]):
			cols.append(fits.Column(name = 'Coeff{}'.format(cf), format='E', array=coeffs[:, cf]))
		self.poly_params = fits.BinTableHDU.from_columns(cols, name='FIT_COEFFS')

	def mean_square_error(self):
		try:
			mse_real = 1 / self.Nfreq * np.nansum((np.real(self.gain_array) - np.real(self.poly_array)) ** 2)
			mse_imag = 1 / self.Nfreq * np.nansum((np.imag(self.gain_array) - np.imag(self.poly_array)) ** 2)
			return max([mse_real, mse_imag])
		except AttributeError:
			print ('Fit Object has no fitting attribute. Fitting of the data is required.')
			return None

	def write_to(self, outfile=None, overwrite=True, **kwargs):
		if outfile is None:
			outfile = self.calfits.replace('.fits', '_poly.fits')
		try:
			with fits.open(self.calfits) as hdus:
				hdus['SOLUTIONS'].data[:, :, :, ::2] = np.real(self.poly_array)
				hdus['SOLUTIONS'].data[:, :, :, 1::2] = np.imag(self.poly_array)
				hdus.append(self.poly_params)
				for key, value in kwargs.items():
					hdus['FIT_COEFFS'].header.insert('TTYPE1', (key, value))
				hdus.writeto(outfile, overwrite=overwrite)
		except AttributeError:
			print ('Fit Object has no fitting attribute. Fitting of the data is required. No file is being written.')
