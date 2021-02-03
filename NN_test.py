import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
matplotlib.style.use("ggplot")
%matplotlib qt

# ______________________ Import data _________________________________________
DATA_DIR = 'DATA/'
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
spectra = pd.concat([pd.DataFrame(spectral_data_raw['flux'].to_list()).iloc[:, 0:950],
                     spectral_data_raw['rave_obs_id']], axis=1).dropna()
spectra_error = pd.concat([pd.DataFrame(spectral_data_raw['flux_error'].to_list()).iloc[0:950],
                           spectral_data_raw['rave_obs_id']], axis=1).dropna()
spectra_all_data = pd.merge(spectra, star_data, on='rave_obs_id', how='inner')
analysis_line1 = pd.read_csv(DATA_DIR + 'analysis_results_line_1.csv')
analysis_line2 = pd.read_csv(DATA_DIR + 'analysis_results_line_2.csv')
analysis_line3 = pd.read_csv(DATA_DIR + 'analysis_results_line_3.csv')

# ______________________ Filters for data ______________________________________

common_ids = np.intersect1d(analysis_line3.rave_obs_id, np.intersect1d(
    analysis_line1.rave_obs_id, analysis_line2.rave_obs_id))

# We keep the same stars for a fair comparison of the algorithm's performance
# Drop the first two columns as they are bypdroducts of the filtering and we don't need them
analysis_line1_cm = analysis_line1[analysis_line1['rave_obs_id'].isin(
    common_ids)].reset_index().drop(['index', 'Unnamed: 0'], axis=1)
analysis_line2_cm = analysis_line2[analysis_line2['rave_obs_id'].isin(
    common_ids)].reset_index().drop(['index', 'Unnamed: 0'], axis=1)
analysis_line3_cm = analysis_line3[analysis_line3['rave_obs_id'].isin(
    common_ids)].reset_index().drop(['index', 'Unnamed: 0'], axis=1)

# ______________________ Preprocess data ______________________________________
# Test-train split
X_train, X_test, y_train, y_test = train_test_split(
    analysis_df.iloc[:, :-1], analysis_df.iloc[:, -1], test_size=0.2,
    random_state=421, shuffle=True)

# Scale the data and remove the ID column
scaler = preprocessing.StandardScaler().fit(X_train.drop('rave_obs_id', axis=1))
X_train_scaled = scaler.fit_transform(X_train.drop('rave_obs_id', axis=1))
X_test_scaled = scaler.fit_transform(X_test.drop('rave_obs_id', axis=1))

# ______________________ Create MPL model with parameters ____________________

mpl_clf = MLPClassifier(solver='lbfgs', alpha=1e-3, max_iter=2500, activation='relu',
                        hidden_layer_sizes=(64, 64, 64), random_state=1)
mpl_clf.fit(X_train_scaled, y_train)
y_pred_mpl = mpl_clf.predict(X_test_scaled)


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements


cm_mpl = confusion_matrix(y_pred_mpl, y_test)
print('Accuracy of MLPClassifier : {}'.format(accuracy(cm_mpl)))


# ________________________ Create Random Forest model w/ parameters __________
forest_clf = RandomForestClassifier(
    n_estimators=1000, criterion='entropy', n_jobs=-1, bootstrap=True, max_depth=10)
forest_clf.fit(X_train_scaled, y_train)
y_pred_forest = forest_clf.predict(X_test_scaled)

cm_forest = confusion_matrix(y_pred_forest, y_test)
print('Accuracy of RandomForestClassifier : {}'.format(accuracy(cm_forest)))

# ________________________ Create SVM w/ parameters __________________________
svm_clf = SVC(kernel='sigmoid', C=.1)
svm_clf.fit(X_train_scaled, y_train)
y_pred_svm = svm_clf.predict(X_test_scaled)

cm_svm = confusion_matrix(y_pred_svm, y_test)
print('Accuracy of SVM : {}'.format(accuracy(cm_svm)))


# _____________ Create model with flux values for classification _____________
# Let's predict the temperature from the spectra
regression_spectra = spectra_all_data.iloc[:, 0:955].drop(
    ['rave_obs_id', 'spectrum_fits', 'hrv_sparv', 'hrv_error_sparv'], axis=1)

regression_spectra = spectra.drop('rave_obs_id', axis=1)
regression_spectra['teff'] = star_data['teff_cal_madera']
regression_spectra = regression_spectra.dropna()

X_train, X_test, y_train, y_test = train_test_split(
    regression_spectra.iloc[:, :-
                            1], regression_spectra.iloc[:, -1], test_size=0.25,
    random_state=421, shuffle=True)

regr = MLPRegressor(hidden_layer_sizes=(256, 512, 512, 256),
                    random_state=1, max_iter=2500).fit(X_train, y_train)
y_pred = regr.predict(X_test)
