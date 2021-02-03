import seaborn as sns
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import pandas as pd

matplotlib.style.use("ggplot")
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['axes.labelsize'] = 22
# matplotlib.rcParams['ylabel.labelsize'] = 18
%matplotlib qt


# ________________________ Import data __________________________________________
DATA_DIR = 'DATA/'
PLOT_DIR = 'figures/'
WAVE_TOP = 8410  # Angstrom
WAVE_BOTTOM = 8790
WAVERANGE = np.linspace(WAVE_TOP, WAVE_BOTTOM, 1000)
star_data = pd.DataFrame(fits.getdata(DATA_DIR + 'data_stars.fits', 1))
# spectra = pd.read_csv(DATA_DIR + 'spectra_fits_raw.csv').iloc[:, :-1]
spectral_data_raw = pd.DataFrame(np.load(
    'spectra_fits_raw.npy', allow_pickle=True),
    columns=['flux', 'flux_error', 'rave_obs_id'])

# Filter those rows that have more flux values than 1000 and separate
# for an easier handling
spectra = pd.concat([pd.DataFrame(spectral_data_raw['flux'].to_list()).iloc[:, 0:1000],
                     spectral_data_raw['rave_obs_id']], axis=1)
spectra_error = pd.concat([pd.DataFrame(spectral_data_raw['flux_error'].to_list()).iloc[0:1000],
                           spectral_data_raw['rave_obs_id']], axis=1)
analysis_line1 = pd.read_csv(DATA_DIR + 'analysis_results_line_1.csv')
analysis_line2 = pd.read_csv(DATA_DIR + 'analysis_results_line_2.csv')
analysis_line3 = pd.read_csv(DATA_DIR + 'analysis_results_line_3.csv')
# Define line parameters for the plots
edges = [8482.5, 8502.5, 8527, 8547, 8650, 8670]

# ________________________ Generate tables ___________________________________
# Generate exploration table 1
data_table = star_data.iloc[0:50, :].drop(
    ['rave_obs_id', 'spectrum_fits'], axis=1)  # Only contain links
data_table['hrv_sparv'] = data_table['hrv_sparv'].astype('float')

# Round the values to the first decimal digit, saves space
print((data_table.round(1)).to_latex())

# Generate table spectra
data_spectra_table = spectra.iloc[0:10, 0:12]
data_spectra_table.columns = [WAVERANGE[0:12].round(2)]
print(data_spectra_table.round(3).to_latex())

# ________________________ Generate plots ____________________________________


# 1. Descrptive analysis: plots, histograms of stars ________________________
i = 100  # Index of the star's flux
params_star = star_data[star_data['rave_obs_id'] == spectra.rave_obs_id[i]]
fig, ax = plt.subplots(1, figsize=[18, 12], tight_layout=True)
ax.plot(WAVERANGE, spectra.iloc[i, 0:1000], lw=1.65, alpha=0.9, c='k')
for edge in edges:
    ax.axvline(edge, ymin=0, ymax=1, lw=1.3, alpha=0.7)
ax.set_xlabel(r'Wavelength [$\AA$]')
ax.set_ylabel(r'Normalized Flux')
plt.savefig(
    PLOT_DIR + 'spectrum_plot_{}_teff{:.0f}.png'.format(params_star['rave_obs_id'].to_list(
    )[0], params_star['teff_cal_madera'].to_list()[0]), dpi=200)

# 2. Correlation plots _________________________________________________________
# Set up the matplotlib figure
sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(12, 18), tight_layout=True)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.triu(np.ones_like(star_data.iloc[:, 2:-6].corr(), dtype=bool))
# Draw the heatmap with the mask and correct aspect ratio
sns.set(font_scale=1)
b = sns.heatmap(star_data.iloc[:, 2:-6].corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
b.set_yticklabels(list(star_data.iloc[:, 2:-6].columns), size=18)
b.set_xticklabels(list(star_data.iloc[:, 2:-6].columns), size=18)
plt.savefig(PLOT_DIR + 'heatmap_stellarparams.png', dpi=200)


# 3. Histogram(s) ______________________________________________________________
params_hists = ['teff_cal_madera', 'logg_cal_madera',
                'm_h_cal_madera', 'hrv_sparv', 'snr_madera', 'flag2_class']
labels_hists = [r'$T_{\mathrm{eff}}$ [K]',
                r'$log \, g$', r'Metallicity [Me/H]',
                r'Heliocentric Radial Velocity [$km/s$]',
                r'Signal-to-Noise Ratio', r'Rave Class']

param_label_index = 0
fig, ax = plt.subplots(3, 2, figsize=[12, 18], tight_layout=True)
for i in range(3):
    for j in range(2):
        param = params_hists[param_label_index]
        label = labels_hists[param_label_index]
        if param == 'flag2_class':
            bin = 16
        else:
            bin = 150

        ax[i, j].hist(star_data[param], bins=bin,
                      histtype='stepfilled', color='k', alpha=0.75, lw=2)
        ax[i, j].set_xlabel(label)
        param_label_index += 1
plt.savefig(PLOT_DIR + 'param_hists.png', dpi=200)


# 4. t-SNE of all data, what do I see? _________________________________________
spectra_tsne = spectra.iloc[:, :950].dropna()
tsned_data = TSNE(perplexity=30).fit_transform(spectra_tsne)
fig, ax = plt.subplots(1, figsize=[15, 15])
ax.scatter(tsned_data[:, 0], tsned_data[:, 1])

# 5. Histograms from the scores ________________________________________________
scores = pd.read_csv(DATA_DIR + '200RFC_iterations_scores.csv')
scores = scores.fillna(0)

fig, ax = plt.subplots(1, figsize=[15, 10], tight_layout=True)
ax.hist(scores['1'][scores['1'] != 0], bins=13,
        alpha=0.75, lw=3, label='Line 1')
ax.hist(scores['2'][scores['2'] != 0], bins=13,
        alpha=0.75, lw=3, label='Line 2')
ax.hist(scores['3'][scores['3'] != 0], bins=13,
        alpha=0.75, lw=3, label='Line 3')
ax.legend(fontsize='x-large')
ax.set_xlabel('Classification Score')
plt.savefig(PLOT_DIR + 'hists_200_iterations.png', dpi=200)

# 6. Line fitting cont and R2 hists  ____________________________________________

fig, ax = plt.subplots(1, 2, figsize=[20, 10], tight_layout=True)
ax[0].hist(analysis_line1.cont_level, label='Line 1', alpha=0.8)
ax[0].hist(analysis_line2.cont_level, label='Line 2', alpha=0.8)
ax[0].hist(analysis_line3.cont_level, label='Line 3', alpha=0.8)
ax[0].legend(fontsize='x-large')
ax[0].set_xlabel('Continuum Level Fit')
ax[1].hist(analysis_line1.r2_score, label='Line 1', alpha=0.8)
ax[1].hist(analysis_line2.r2_score, label='Line 2', alpha=0.8)
ax[1].hist(analysis_line3.r2_score, label='Line 3', alpha=0.8)
ax[1].legend(fontsize='x-large')
ax[1].set_xlabel('R2 Score Fit')
plt.savefig(PLOT_DIR + 'hists_lineanalysis.png', dpi=200)

# 7. Results for the   ____________________________________________
