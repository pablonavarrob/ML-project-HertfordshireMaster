import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from functools import reduce
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import wofz
from sklearn.metrics import mean_squared_error
matplotlib.style.use("ggplot")
%matplotlib qt

# Import data
DATA_DIR = 'DATA/'
WAVE_TOP = 8410  # Angstrom
WAVE_BOTTOM = 8790
WAVERANGE = np.linspace(WAVE_TOP, WAVE_BOTTOM, 1000)
star_data = fits.getdata(DATA_DIR + 'data_stars.fits', 1, memmap=False)
spectra = pd.read_csv(DATA_DIR + 'spectra_fits.csv').iloc[:, :-1]

#________________ Define functions ____________________________________________#


def cauchy(x):
    profile = (1 / np.pi) * (1 / (1 + x**2))
    return profile


def gauss(x):
    profile = (1 / np.sqrt(2 * np.pi)) * np.e**(-(1 / 2) * x**2)
    return profile


def spectrum_c(wavelength, C1, C2, C3, C4, C5):
    """ Lorentzian (Cauchy). """
    f = C1 + C2 * (wavelength - waveref) + C3 * cauchy((wavelength - C4) / C5)
    return f


def spectrum_g(wavelength, C1, C2, C3, C4, C5):
    """ Gaussian."""
    f = C1 + C2 * (wavelength - waveref) + C3 * gauss((wavelength - C4) / C5)
    return f


def get_line_info(line_number, all=False):

    left_line_1 = 8482.5
    right_line_1 = 8502.5
    left_line_2 = 8527
    right_line_2 = 8547
    left_line_3 = 8650
    right_line_3 = 8670

    # Get in aray for easier handling
    line_cut_refs = [(left_line_1, right_line_1),
                     (left_line_2, right_line_2),
                     (left_line_3, right_line_3)]

    if all == True:
        return line_cut_refs
    elif (all == False and line_number == 1):
        return line_cut_refs[0]
    elif (all == False and line_number == 2):
        return line_cut_refs[1]
    elif (all == False and line_number == 3):
        return line_cut_refs[2]


def get_masks_lines(waverange):

    line_cut_refs = get_line_info(None, True)
    # Define continuum masks
    mask_line_1 = ((waverange > line_cut_refs[0][0]) &
                   (waverange < line_cut_refs[0][1]))

    mask_line_2 = ((waverange > line_cut_refs[1][0]) &
                   (waverange < line_cut_refs[1][1]))

    mask_line_3 = ((waverange > line_cut_refs[2][0]) &
                   (waverange < line_cut_refs[2][1]))

    return [mask_line_1, mask_line_2, mask_line_3]


def get_continuum(waverange_, flux_):
    """ Enter the line info as a list/array with touples containing
    the left and right wavelength values from which the line will be masked. """

    # Import line data and pertinent corrections to NaN values
    flux = flux_.fillna(0)
    # Mask to remove 0 from the spectra
    mask_nozero = np.array(flux != 0)
    waverange = waverange_[mask_nozero]
    flux = flux[mask_nozero]

    mask_lines = get_masks_lines(waverange)
    # Combine masks into one to avoid value overlap: this masks the continuum
    mask_lines_final = np.array([any(tup) for tup in zip(mask_lines[0],
                                                         mask_lines[1],
                                                         mask_lines[2])])

    return waverange[~mask_lines_final], flux[~mask_lines_final]


def get_filtered_line(waverange_, flux_, line_number):
    """ Returns the selected line without the other two for proper fitting. """
    # Import line data and pertinent corrections to NaN values
    flux = flux_.fillna(0)
    # Mask to remove 0 from the spectra
    mask_nozero = np.array(flux != 0)
    waverange = waverange_[mask_nozero]
    flux = flux[mask_nozero]

    mask_line = get_masks_lines(waverange)[line_number - 1]

    # mask_lines = get_masks_lines(waverange)
    # del mask_lines[line_number - 1]
    #
    # mask_lines_final = np.array([any(tup) for tup in zip(mask_lines[0],
    #                                                      mask_lines[1])])

    return waverange[mask_line], flux[mask_line]


#________________ Test plots __________________________________________________#
left_line_1 = 8480
right_line_1 = 8505
left_line_2 = 8525
right_line_2 = 8550
left_line_3 = 8645
right_line_3 = 8675

# Get in aray for easier handling
line_cut_refs = [(left_line_1, right_line_1),
                 (left_line_2, right_line_2),
                 (left_line_3, right_line_3)]

fig, ax = plt.subplots(1, figsize=[15, 10])
ax.axvline(right_line_1, 0, 1.1, c='k', alpha=0.75)
ax.axvline(left_line_1, 0, 1.1, c='k', alpha=0.75)
ax.axvline(right_line_2, 0, 1.1, c='k', alpha=0.75)
ax.axvline(left_line_2, 0, 1.1, c='k', alpha=0.75)
ax.axvline(right_line_3, 0, 1.1, c='k', alpha=0.75)
ax.axvline(left_line_3, 0, 1.1, c='k', alpha=0.75)
ax.plot(WAVERANGE[0:1000], spectra.iloc[123, 0:1000])

#________________ First guesses for the curve_fit _____________________________#
i = 123
line_boundaries = get_line_info(1)
ref_wave = sum(line_boundaries) / 2
waveref = ref_wave
contwave, contflux = get_continuum(WAVERANGE, spectra.iloc[i])

# 1. Flux of the reference wavelenght (interval half)
# searchsorted finds the index where the value would be place in the sorted array
theta1_g = spectra.iloc[i][np.searchsorted(WAVERANGE, ref_wave, side='right')]
# 2. Continuum level
theta2_g = np.polyfit(contwave, contflux, deg=1)[0]
# 3. Strength of the absorption line
theta3_g = theta3_g = np.min(spectra.iloc[i][(
    WAVERANGE > line_boundaries[0]) & (WAVERANGE < line_boundaries[1])])
# 4. Wavelength of the peak
mask_peak = ((WAVERANGE > line_boundaries[0]) &
             (WAVERANGE < line_boundaries[1]))
theta4_g = WAVERANGE[spectra.iloc[i] == np.min(spectra.iloc[i][mask_peak])][0]
# 5. Line width
theta5_g = line_boundaries[1] - line_boundaries[0]

guess = ([theta1_g, theta2_g, theta3_g, theta4_g, theta5_g])

line_w, line_f = get_filtered_line(WAVERANGE, spectra.iloc[i], 1)
chi2_minc, chi2_minc_covar = curve_fit(
    spectrum_g, line_w, line_f, p0=guess, maxfev=2000)
