import os
import sys
import glob
import gc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
matplotlib.style.use("ggplot")
%matplotlib qt

DATA_DIR = 'DATA/'
WAVE_TOP = 8410  # Angstrom
WAVE_BOTTOM = 8790
WAVERANGE = np.linspace(WAVE_TOP, WAVE_BOTTOM, 1000)

# Need to load all spectra files into a single, accessible .csv file that
# contains the flux values for each spectrum.
star_data = fits.open(DATA_DIR + "data_stars.fits", memmap=True)[1].data


def parse_fitsfilename(filename):
    """ Extracts the star id from the filename for future referencing
    and cross-match with the star_data dataset.

    From comparison with star_data, all the numbers and letters
    from RAVE_ until file extension are part of the feature
    with the name rave_obs_id. """

    return filename.replace(DATA_DIR + 'fits/RAVE_', '').replace('.fits', '')


def get_fits_data(fits_dir):

    hdul = fits.open(fits_dir, memmap=False)
    image_data = [hdul[1].data, hdul[2].data]
    hdul.close()
    gc.collect()

    return image_data


def read_spectralfits():
    """ Routine to read all spetra fits from a file and store them in a
    separate array for easier handling. """

    spectra = []
    rave_obs_id = []
    spectra_errors = []
    errors = []
    i = 0
    print('Reading fits files...')
    for f in glob.glob(DATA_DIR + "fits/*.fits"):
        i += 1
        try:
            spectrum = get_fits_data(f)
            spectra.append(spectrum[0])
            spectra_errors.append(spectrum[1])
            rave_obs_id.append(parse_fitsfilename(f))

        except:
            print("There was an unexpected error in file {}".format(f))
            errors.append(parse_fitsfilename(f))
            pass

        if i % 1000 == 0:
            print('{:.0} done'.format(i / 100))

    print('Fits succesfully read, storing in .csv file...')
    data = {'flux': spectra, 'flux_error': spectra_errors,
            'rave_obs_id': rave_obs_id}
    df_spectra = pd.DataFrame(data)
    df_spectra.to_csv(DATA_DIR + 'spectra_fits_raw.csv', index=False)
    np.save(DATA_DIR + 'spectra_fits_raw.npy', df_spectra.to_numpy())
    (pd.DataFrame(errors)).to_csv(DATA_DIR + 'corrupted_files.csv', index=False)


if __name__ == '__main__':
    read_spectralfits()
