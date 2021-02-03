import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from sklearn.metrics import r2_score
%matplotlib qt

matplotlib.style.use("ggplot")
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18
matplotlib.rcParams['axes.labelsize'] = 22

# ______________________ Import data _________________________________________
DATA_DIR = 'DATA/'
PLOT_DIR = 'figures/'
WAVE_TOP = 8410  # Angstrom
WAVE_BOTTOM = 8790
WAVERANGE = np.linspace(WAVE_TOP, WAVE_BOTTOM, 1000)
star_data = pd.DataFrame(fits.getdata(DATA_DIR + 'data_stars.fits', 1))
spectra = pd.read_csv(DATA_DIR + 'spectra_fits_raw.csv').iloc[:, :-1]
spectral_data_raw = pd.DataFrame(np.load(
    'spectra_fits_raw.npy', allow_pickle=True),
    columns=['flux', 'flux_error', 'rave_obs_id'])

# Filter those rows that have more flux values than 1000 and separate
# for an easier handling
spectra = pd.concat([pd.DataFrame(spectral_data_raw['flux'].to_list()).iloc[:, 0:950],
                     spectral_data_raw['rave_obs_id']], axis=1).dropna()
spectra_error = pd.concat([pd.DataFrame(spectral_data_raw['flux_error'].to_list()).iloc[0:950],
                           spectral_data_raw['rave_obs_id']], axis=1).dropna()
spectra_all_data = pd.merge(spectra, star_data, on='rave_obs_id', how='inner')


# ________________________ Model _______________________________________________

def Lorentz(x, y0, a, x0, gamma):
    """ Lorentzian profile: y0 is the offset, x0 is the wavelength at the peak,
    a is a scaling factor and gamma is a parameter. """

    return y0 + a * (1 / (np.pi)) * (1 + ((x - x0) / gamma)**2)**-1


def get_line_info(line_number, all=False):

    left_line_1 = 8484
    right_line_1 = 8501
    left_line_2 = 8529
    right_line_2 = 8545
    left_line_3 = 8652
    right_line_3 = 8668

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

# ________________________ Statistical analysis ________________________________


def fit_line(spectra, line_number, n, get_plot=False, print_report=False):
    """ Selects one calcium line and, dynamically selects the it
    from the lowest flux value and a small range round it for fitting. """

    if line_number == 1:
        rng = 3  # Angstroms around central wavelengths
    elif line_number == 2:
        rng = 5  # Angstroms around central wavelengths
    elif line_number == 3:
        rng = 4.  # Angstroms around central wavelengths

    line_ranges = get_line_info(line_number)
    waverange = WAVERANGE[0:len(spectra)]
    line_mask = ((waverange > line_ranges[0]) & (waverange < line_ranges[1]))
    peak_wavelength = (waverange[line_mask][spectra[line_mask]
                                            == np.min(spectra[line_mask])])[0]

    # 2. Get the closest points to mask for curve_fit
    mask_peak = ((waverange > peak_wavelength - rng)
                 & (waverange < peak_wavelength + rng))
    peak_waverange = waverange[mask_peak]
    peak_fluxes = spectra[mask_peak]

    mean = sum(peak_waverange * peak_fluxes) / sum(peak_fluxes)
    sigma = np.sqrt(sum(peak_fluxes *
                        (peak_waverange - mean)**2) / sum(peak_fluxes))
    guess = [1, min(peak_fluxes), mean, sigma]

    try:
        poptl, pcovl = curve_fit(Lorentz, xdata=peak_waverange, ydata=peak_fluxes,
                                 p0=guess, maxfev=2500)

    except:
        print('Fitting did not converge.')
        poptl = [0, 0, 0, 0]
        r2 = 0

        return poptl, r2, guess

    if get_plot:
        fig, ax = plt.subplots(1, figsize=[15, 10], tight_layout=True)
        ax.plot(waverange, spectra, c='k')
        ax.plot(waverange, Lorentz(waverange, *poptl), c='r')
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel(r'Normalized Flux')
        # ax.plot(calcium_lineII_wave, Voigt(calcium_lineII_wave, *poptv), c='orange')
        plt.savefig(
            PLOT_DIR + 'spectra_fit_line{}_number{}.png'.format(line_number, n), dpi=200)

    # Compute sum of squared errors
    r2 = r2_score(peak_fluxes, Lorentz(peak_waverange, *poptl))
    # print("Mean Squared error of line {} is {:.3}".format(line_number, mse))

    if print_report:
        print(tabulate([['Continuum Level', '{}'.format(
            guess[0]), '{:.3}'.format(poptl[0])],
            ['Flux Peak', '{:.4}, fixed'.format(
                guess[1]), '{:.4}'.format(poptl[1])],
            ['Wavelength Peak', '{:.4}'.format(
                guess[2]), '{:.4}'.format(poptl[2])],
            ['Width Line', '{:.4}'.format(
                guess[3]), '{:.4}'.format(poptl[3])],
            ['R2 Score', '', '{:.4}'.format(r2)]],
            ['Parameter', 'Guess', 'Fit'], tablefmt='simple',
            numalign='left'))

    return poptl, r2, guess


def statistical_analyis(line_number, spectra, star_data):
    """ Performs the modeling for a selected line. Outputs the defined
    dataframe and stores it. """

    # Iteratively analyze the third line of the first 1000 spectra.
    results_statistical_analysis = []
    for i in range(len(spectra)):
        poptl, r2, guess = fit_line(spectra.iloc[i, :-1], line_number, i)
        results_statistical_analysis.append({'cont_level': poptl[0],
                                             'flux_peak': poptl[1],
                                             'peak_wave': poptl[2],
                                             'width_line': abs(poptl[3]),
                                             'guess_width': guess[3],
                                             'r2_score': r2})

    results_statistical_analysis = pd.DataFrame(results_statistical_analysis)
    star_data_test = star_data
    spectra_test = spectra

    # 1. Create data set for the analysis
    analysis_df = pd.DataFrame(data={'cont_level': results_statistical_analysis.cont_level,
                                     'peak_wave': results_statistical_analysis.peak_wave,
                                     'peak_flux': results_statistical_analysis.flux_peak,
                                     'width_line': results_statistical_analysis.width_line,
                                     'r2_score': results_statistical_analysis.r2_score,
                                     'teff': star_data_test.teff_cal_madera,
                                     'logg': star_data_test.logg_cal_madera,
                                     'm_h': star_data_test.m_h_cal_madera,
                                     'hrv': star_data_test.hrv_sparv,
                                     'snr': star_data_test.snr_madera,
                                     'flag2_class': star_data_test.flag2_class,
                                     'rave_obs_id': spectra['rave_obs_id']})

    analysis_df.flag2_class = pd.Categorical(analysis_df.flag2_class)
    analysis_df['code'] = analysis_df.flag2_class.cat.codes
    analysis_df = analysis_df.drop('flag2_class', axis=1)

    # Remove data that could interfere with the analysis due to problematic
    # fitting and also missing values.
    r2_threshold_mask = ((analysis_df['r2_score'] > 0.875) &
                         (analysis_df['cont_level'] > 0.925) &
                         (analysis_df['cont_level'] < 1.075) &
                         (analysis_df['code'] != 0))
    analysis_df = analysis_df[r2_threshold_mask].dropna()
    analysis_df.to_csv(
        DATA_DIR + 'analysis_results_line_{}.csv'.format(line_number), index=False)

    return analysis_df
