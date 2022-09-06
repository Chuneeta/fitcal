from setuptools import setup
from polyFit import version
import os
import json

data = [version.git_origin, version.git_hash,
		version.git_description, version.git_branch]
with open(os.path.join('polyFit', 'GIT_INFO'), 'w') as outfile:
	json.dump(data, outfile)


def package_files(package_dir, subdirectory):
	# walk the input package_dir/subdirectory
	# return a package_data list
	paths = []
	directory = os.path.join(package_dir, subdirectory)
	for (path, directories, filenames) in os.walk(directory):
		for filename in filenames:
			path = path.replace(package_dir + '/', '')
			paths.append(os.path.join(path, filename))
	return paths


data_files = package_files('polyFit', 'data')

setup_args = {
	'name':         'polyFit',
	'author':       'Chuneeta Nunhokee',
	'url':          'https://github.com/Chuneeta/polyFit',
	'license':      'BSD',
	'version':      version.version,
	'description':  'MWA Calibration Polynomial Fitting',
	'packages':     ['polyFit'],
	'package_dir':  {'polyFit': 'polyFit'},
	'package_data': {'polyFit': data_files},
	'install_requires': ['numpy>=1.16.5','matplotlib>=2.2', 'pytest'],
	'include_package_data': True,
	'zip_safe':     False,
	'scripts': ['scripts/run_polyfit.py']
}

if __name__ == '__main__':
	setup(*(), **setup_args)
