import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
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
star_data.flag2_class = star_data.flag2_class.astype('category')
categories = dict(enumerate(star_data.flag2_class.cat.categories))
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
# Drop the first column: bypdroducts of the filtering and we don't need them
analysis_line1_cm = analysis_line1[analysis_line1['rave_obs_id'].isin(
    common_ids)].reset_index().drop('index', axis=1)
analysis_line2_cm = analysis_line2[analysis_line2['rave_obs_id'].isin(
    common_ids)].reset_index().drop('index', axis=1)
analysis_line3_cm = analysis_line3[analysis_line3['rave_obs_id'].isin(
    common_ids)].reset_index().drop('index', axis=1)

# ______________________ ML hyperparameter selection _________________________
n_estimators = [100, 250, 500, 1000, 1500]
max_depth = [10, 25, 50, 100, None]
min_samples_split = [2, 5, 10, 15]
min_samples_leaf = [2, 5, 10, 15]

i = analysis_line3_cm
X_train_hyperparams, X_test_hyperparams, y_train_hyperparams, y_test_hyperparams = train_test_split(
    i.iloc[:, :-1], i.iloc[:, -1], test_size=0.3,
    random_state=421, shuffle=True)
scaler = preprocessing.StandardScaler().fit(
    X_train_hyperparams.drop('rave_obs_id', axis=1))
X_train_scaled_hyperparams = scaler.fit_transform(
    X_train_hyperparams.drop('rave_obs_id', axis=1))
X_test_scaled_hyperparams = scaler.fit_transform(
    X_test_hyperparams.drop('rave_obs_id', axis=1))

exploration_results = []
i = 0
for est in n_estimators:
    for depth in max_depth:
        for split in min_samples_split:
            for leaf in min_samples_leaf:
                forest_clf = RandomForestClassifier(
                    n_estimators=est, max_depth=depth, n_jobs=-1,
                    min_samples_split=split, min_samples_leaf=leaf, bootstrap=True)
                forest_clf.fit(X_train_scaled_hyperparams, y_train_hyperparams)
                y_pred_hyperparams = forest_clf.predict(
                    X_test_scaled_hyperparams)
                exploration_results.append({'n_estimators': est,
                                            'max_depth': depth,
                                            'min_samples_split': split,
                                            'min_samples_leaf': leaf,
                                            'score': accuracy_score(y_test_hyperparams,
                                                                    y_pred_hyperparams)})
                print('Step {} done'.format(i))
                i += 1

exploration_results = pd.DataFrame(exploration_results)
exploration_results.to_csv(
    DATA_DIR + "randomforests_explorationresults.csv", index=False)


# ______________________ ML models ___________________________________________
exploration_results = pd.read_csv(
    DATA_DIR + "randomforests_explorationresults.csv")
best_mode = (exploration_results[exploration_results.score == np.max(
    exploration_results.score)].iloc[0]).astype(int)

# Test recursively the
predicted_labels = []
true_labels = []
j = 0
for i in [analysis_line1_cm, analysis_line2_cm, analysis_line3_cm]:
    X_train, X_test, y_train, y_test = train_test_split(
        i.iloc[:, :-1], i.iloc[:, -1], test_size=0.3,
        random_state=421, shuffle=True)
    # Scale the data and remove the ID column
    scaler = preprocessing.StandardScaler().fit(X_train.drop('rave_obs_id', axis=1))
    X_train_scaled = scaler.fit_transform(
        X_train.drop('rave_obs_id', axis=1))
    X_test_scaled = scaler.fit_transform(
        X_test.drop('rave_obs_id', axis=1))

    # Create the classifier, train it and predict
    forest_clf = RandomForestClassifier(
        n_estimators=best_mode.n_estimators, max_depth=best_mode.max_depth,
        min_samples_split=best_mode.min_samples_split, min_samples_leaf=best_mode.min_samples_leaf,
        n_jobs=-1, bootstrap=True)
    forest_clf.fit(X_train_scaled, y_train)
    y_pred_forest = forest_clf.predict(X_test_scaled)
    predicted_labels.append(y_pred_forest)
    true_labels.append(y_test)
    del scaler
    del forest_clf

    j += 1
    print('Accuracy of RF Classification on line {}: {}'.format(
        j, accuracy_score(y_test, y_pred_forest)))


# ______________________ ML model for all lines ________________________________
# Merge data sets for a compound one
analysis_all_lines = pd.merge(analysis_line1_cm, pd.merge(
    analysis_line2_cm, analysis_line2_cm, on='rave_obs_id'), on='rave_obs_id')
# Clean the recently created data set
analysis_all_lines['cont_level_avg'] = (pd.DataFrame(
    [analysis_all_lines.cont_level,
     analysis_all_lines.cont_level_x,
     analysis_all_lines.cont_level_y]).transpose()).mean(axis=1)
# Remove items not necessary
removal_list_all = ['r2_score', 'teff_',
                    'logg_', 'm_h_', 'hrv_', 'snr_', 'code_']
for item in removal_list_all:
    analysis_all_lines = analysis_all_lines.loc[:,
                                                ~analysis_all_lines.columns.str.startswith(item)]
# Perform analysis 200 times
all_rf_test_scores = []
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    analysis_all_lines.drop('code', axis=1), analysis_all_lines.code, test_size=0.3,
    random_state=421, shuffle=True)
# Scale the data and remove the ID column
scaler = preprocessing.StandardScaler().fit(
    X_train_all.drop('rave_obs_id', axis=1))
X_train_scaled_all = scaler.fit_transform(
    X_train_all.drop('rave_obs_id', axis=1))
X_test_scaled_all = scaler.fit_transform(
    X_test_all.drop('rave_obs_id', axis=1))
j = 0
for i in range(200):
    # Create the classifier, train it and predict
    forest_clf = RandomForestClassifier(
        n_estimators=100, max_depth=15,
        min_samples_split=10, min_samples_leaf=10,
        n_jobs=-1, bootstrap=True)
    forest_clf.fit(X_train_scaled_all, y_train_all)
    y_pred_forest_all = forest_clf.predict(X_test_scaled_all)
    all_rf_test_scores.append(accuracy_score(y_test_all, y_pred_forest))
    print(j)
    j += 1
