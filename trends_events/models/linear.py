import numpy as np
from scipy.optimize import curve_fit


def linear(x, m, c):
	return m*x+c


class Linear:
	def __init__(self):
		self.VLARGE = 5000

	def predict(self, data, confs, gap=0, predtill=1):
		assert predtill-1 <= gap
		true = data[:, -predtill:, :]
		pred = []

		for i in range(data.shape[2]):
			trend = data[0, :-1-gap, i]
			conf = confs[0, :-1-gap, i]

			errs = []
			errf = self.VLARGE
			fits = []

			weeks = list(range(trend.shape[0]))
			try:
				params, _ = curve_fit(linear, range(len(weeks)), trend, method='lm', p0=(0, 0), sigma=conf)
				err_l = np.sqrt(np.sum(np.square(np.array([linear(tmp, params[0], params[1]) for tmp in range(len(weeks))])-trend)))
				errs.append(err_l)
				fits.append(params)
				if err_l < errf:
					errf = err_l
					pred_tmp = [linear(tmp, params[0], params[1]) for tmp in range(data.shape[1]-predtill, data.shape[1])]
			except:
				raise
			if i%100 == 0:
				print("Done", i, "/", data.shape[2])
			pred.append(pred_tmp)
		pred = np.expand_dims(np.array(pred).T, axis=0)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
