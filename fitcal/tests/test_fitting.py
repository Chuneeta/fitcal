from polyFit.fitting import Fit
from numpy.polynomial import Polynomial
import numpy as np
import unittest


calfile = '/Users/ridhima/Documents/mwa/calibration/mwa_qa/test_files/test_1061315688.fits'

class TestFit(unittest.TestCase):
	def test__init__(self):
		f = Fit(calfile)
		self.assertEqual(f.calfits, calfile)
		self.assertEqual(f.gain_array.shape, (1, 128, 768, 4))
		np.testing.assert_almost_equal(np.real(f.gain_array[0, 0, 10, :]), 
		np.array([0.06321781, 0.00887845, 0.03955971, 0.27421103]))
		self.assertEqual(f.convergence.shape, (1, 768))
		self.assertEqual(f.convergence[0, 10], 5.029599444669263e-07)
		self.assertEqual(f.Ntimes, 1)
		self.assertEqual(f.Nants, 128)
		self.assertEqual(f.Nfreq, 768)
		self.assertEqual(f.Npols, 4)
		self.assertEqual(f.Nchan, 32)
		self.assertEqual(f.Nband, 24)

	def test_unflagged_channels(self):
		f = Fit(calfile)
		unflagged_channels = f.unflagged_channels(edge_width=2, clip_width=0)
		self.assertEqual(len(unflagged_channels), 648)
		self.assertEqual(unflagged_channels[0], 2)
		self.assertEqual(unflagged_channels[-1], 765)	

	def test_polynomial_fit(self):
		f = Fit(calfile)
		inds = np.where(~np.isnan(f.gain_array[0, 10, :, 0]))[0]
		poly = f._polynomial_fit(np.arange(f.Nfreq)[inds], f.gain_array[0, 10, inds, 0], 3, weights=None)
		self.assertEqual(type(poly), tuple)
		self.assertTrue(isinstance(poly[0], Polynomial))
		self.assertTrue(isinstance(poly[1], Polynomial))
	
	def test_fit_data(self):
		f = Fit(calfile)
		f.fit_data(3)
		self.assertEqual(f.poly_array.shape, (1, 128, 768, 4))
		self.assertEqual(f.poly_array.dtype, 'complex128')
		np.testing.assert_almost_equal(np.real(f.poly_array[0, 0, 10, :]), 
		np.array([0.01208641, 0.01098337, 0.01217154, 0.28863853]))
		inds = np.where(~np.isnan(f.gain_array[0, 0, :, 0]))[0]
		poly = f._polynomial_fit(np.arange(f.Nfreq)[inds], f.gain_array[0, 0, inds, 0], 3, weights=None)
		self.assertEqual(f.poly_params.data[0][0], 0)
		self.assertEqual(f.poly_params.data[0][1], 0)
		self.assertEqual(f.poly_params.data[0][2], 'XX')
		self.assertEqual(f.poly_params.data[0][3], 'REAL')
		np.testing.assert_almost_equal(f.poly_params.data[0][4], poly[0].coef[0])
		np.testing.assert_almost_equal(f.poly_params.data[0][5], poly[0].coef[1])
		np.testing.assert_almost_equal(f.poly_params.data[0][6], poly[0].coef[2])
		np.testing.assert_almost_equal(f.poly_params.data[0][7], poly[0].coef[3])
		np.testing.assert_almost_equal(f.poly_params.data[1][4], poly[1].coef[0])
		np.testing.assert_almost_equal(f.poly_params.data[1][5], poly[1].coef[1])
		np.testing.assert_almost_equal(f.poly_params.data[1][6], poly[1].coef[2])
		np.testing.assert_almost_equal(f.poly_params.data[1][7], poly[1].coef[3])
		
	def test_mean_square_error(self):
		f = Fit(calfile)
		f.fit_data(3)
		mse = f.mean_square_error()
		self.assertEqual(mse, 0.4087536776025334)
		f = Fit(calfile)
		mse =  f.mean_square_error()
		self.assertTrue(mse == None)