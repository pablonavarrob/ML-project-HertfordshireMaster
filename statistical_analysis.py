import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from sklearn import manifold
from sklearn.decomposition import PCA
matplotlib.style.use("ggplot")

DATA_DIR = 'DATA/'
WAVE_TOP = 8430  # Angstrom
WAVE_BOTTOM = 8740

star_data = fits.getdata(DATA_DIR + 'data_stars.fits', 1, memmap=False)
spectra = pd.read_csv(DATA_DIR + 'spectra_fits.csv')
waverange = np.linspace(WAVE_TOP, WAVE_BOTTOM, 960)

# Try runnning PCA to find outliers and remove them from the data set, try
# using PCA again
pca_spectra = PCA(n_components=2)
pcad_spectra = pca_spectra.fit_transform(spectra.iloc[:, 0:900])
mask_correct = ((pcad_spectra[:, 0] < 2) &
                (pcad_spectra[:, 0] > -2) &
                (pcad_spectra[:, 1] < 1) &
                (pcad_spectra[:, 1] > -1.55))
outliers = pcad_spectra[~mask_correct]

# Remove outliers from the spectral data set and run PCA again
pca_spectra_nooutliers = PCA(n_components=2)
pcad_spectra_nooutliers = pca_spectra_nooutliers.fit_transform(
    spectra.iloc[:, 0:900][mask_correct])

# Try tSNE
tsne = manifold.TSNE(n_components=2, perplexity=15)
tsned_results = tsne.fit_transform(spectra.iloc[:, 0:900][mask_correct])

# Plot the results from the two first components
fig, ax = plt.subplots(1, figsize=[15, 15])
ax.scatter(pcad_spectra_nooutliers[:, 0],
           pcad_spectra_nooutliers[:, 1],
           c=star_data['teff_madera'][mask_correct])
plt.show()

fig, ax = plt.subplots(1, figsize=[15, 15])
ax.scatter(tsned_results[:, 0],
           tsned_results[:, 1],
           c=star_data['snr_madera'][mask_correct])
plt.show()
