from argparse import ArgumentParser
from fitcal.gain_fitting import HyperSolnFits
import numpy as np

parser = ArgumentParser(
    description="Polynomial Fitting of the calibration solutions")
parser.add_argument('soln', type=str, help='Hyperdrive fits file')
parser.add_argument('-a', '--amplitude', action='store_true',
                    help='If set, will smooth the amplitudde of the calibration solutions.')
parser.add_argument('-p', '--phase', action='store_true',
                    help='If set, will fit the phase of the calibration solutions.')
parser.add_argument('-n', '--order', dest='order', default=3,
                    type=int, help='Polynomial order to fit for for the amplitudes.')
parser.add_argument('-w', '--window', dest='window', default=99, type=int,
                    help='Window size for the Savitzky-Golay Filter for amplitude smoothing.')
parser.add_argument('-o', '--outfile', dest='outfile', default=None,
                    help='Output file containting the fitted data.')

args = parser.parse_args()

hypersoln = HyperSolnFits(args.soln)
if args.amplitude and args.phase:
    hypersoln.get_fitted_solutions(
        order=args.order, window=args.window, weights=None, fit_amplitude=True, fit_phase=True)

elif args.amplitude and not args.phase:
    hypersoln.get_fitted_solutions(
        order=args.order, window=args.window, weights=None, fit_amplitude=True, fit_phase=False)

elif not args.amplitude and args.phase:
    hypersoln.get_fitted_solutions(
        order=args.order, window=args.window, weights=None, fit_amplitude=False, fit_phase=True)

else:
    hypersoln.get_fitted_solutions(
        order=args.order, window=args.window, weights=None, fit_amplitude=False, fit_phase=False)

# plotting the fitted solutions
hypersoln.plot_solutions(args.outfile)
# saving the solutionss
hypersoln.write_to(args.outfile)
