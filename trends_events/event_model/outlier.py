import numpy as np
from scipy.stats import binom_test

class OutlierDetector:
	def __init__(self):
		pass

	def get_outliers(self, trend, conf, curve, tot):
		ntrend = (trend-curve)
		std = np.std(ntrend)
		outliers = np.array([binom_test(int(trend[i]*tot[i]), tot[i], min(1, curve[i]+std*1.96), alternative='greater') for i in range(trend.shape[0])]) < 0.05

		inds = np.argwhere(outliers)[:, 0]
		scores = [binom_test(int(trend[i]*tot[i]), tot[i], min(1, curve[i]+std*1.96), alternative='greater') for i in inds]
		return outliers, inds, scores
