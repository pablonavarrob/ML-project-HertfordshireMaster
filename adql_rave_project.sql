SELECT
t1.rave_obs_id , t1.spectrum_fits , t2.hrv_sparv , t2.hrv_error_sparv , t3. teff_cal_madera , t3.teff_error_madera , t3.logg_cal_madera , t3. logg_error_madera , t3.m_h_cal_madera , t3.m_h_error_madera , t3. snr_madera , t4.flag1_class , t4.flag2_class , t4.flag3_class , t4.
w1_class , t4.w2_class , t4.w3_class
FROM ravedr6.dr6_spectra AS t1
JOIN ravedr6.dr6_sparv AS t2 ON t1.rave_obs_id = t2.rave_obs_id
JOIN ravedr6.dr6_madera AS t3 ON t1.rave_obs_id = t3.rave_obs_id
JOIN ravedr6.dr6_classification AS t4 ON t1.rave_obs_id = t4.rave_obs_id
WHERE t3.teff_cal_madera between 4750 and 6500 AND t3.snr_madera > 65 ORDER BY t1.rave_obs_id DESC
LIMIT 10000
