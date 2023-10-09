from argparse import ArgumentParser
from pathlib import Path
from polyFit.fitting import Fit
import numpy as np

parser = ArgumentParser(description="Polynomial Fitting of the calibration solutions")
parser.add_argument('soln', type=str, help='Hyperdrive fits file')
parser.add_argument('-s', '--start', dest='start', default=3, type=int, help='Starting polynomial order to fit for. Default is 3.')
parser.add_argument('-n', '--norder', dest='norder', default=7, type=int, help='number of order to iterate over. Defualt is 7.')
parser.add_argument('-o', '--outfile', dest='outfile', default=None, help='Output file containting the fitted data')

args = parser.parse_args()
m = Fit(args.soln)
orders = np.arange(args.start, args.start + args.norder)
errors = []
for order in orders:
    m.fit_data(order)
    mse = m.mean_square_error()
    errors.append(mse)
    print ('Polynomial Order {} : MSE : {}'.format(order, mse))
# finding order with the least error
ind = errors.index(min(errors))
print ('Choosing polynomial order {} with MSE {}'.format(orders[ind], errors[ind]))
m.fit_data(orders[ind])
m.write_to(args.outfile, ORDER=orders[ind], MSE=errors[ind])
