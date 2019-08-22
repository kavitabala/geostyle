import numpy as np
from scipy.optimize import curve_fit
from numpy import exp, sin

def linear(x, m, c):
	return m*x+c

def cyclic(x, m1, w, f, o, b1):
	return (m1*(exp(w*sin(f*x+o))/exp(w)+b1))

class Cyclic:
	def __init__(self):
		self.VLARGE = 5000
		self.explain_factor = 0.8
		self.cycusbounds = ([0, 0, 1/72, -3.14, 0], [1, 20, 1/6, 3.14, 1])
		self.cycdsbounds = ([-1, 0, 1/72, -3.14, -1], [0, 20, 1/6, 3.14, 0])


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
			try:
				params, _ = curve_fit(cyclic, range(len(weeks)), trend, method='trf', p0=(np.max(trend)-np.min(trend), 2, 1/8, 0, np.min(trend)), bounds=self.cycusbounds, sigma=conf)
				m1, k, f, o, b1 = params[0], params[1], params[2], params[3], params[4]
				err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp, m1, k, f, o, b1) for tmp in range(len(weeks))])-trend)))
				errs.append(err1)
				fits.append(params)
				if err1 < errf and err1 < self.explain_factor*err_l:
					errc = err1
					pred_tmp = [cyclic(tmp, m1, k, f, o, b1) for tmp in range(data.shape[1]-predtill, data.shape[1])]
			except KeyboardInterrupt:
				raise
			except:
				pass
			try:
				params, _ = curve_fit(cyclic, range(len(weeks)), trend, method='trf', p0=(np.max(trend)-np.min(trend), 4, 1/8, 0, np.min(trend)), bounds=self.cycusbounds, sigma=conf)
				m1, k, f, o, b1 = params[0], params[1], params[2], params[3], params[4]
				err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp, m1, k, f, o, b1) for tmp in range(len(weeks))])-trend)))
				errs.append(err1)
				fits.append(params)
				if err1 < errf and err1 < self.explain_factor*err_l:
					errc = err1
					pred_tmp = [cyclic(tmp, m1, k, f, o, b1) for tmp in range(data.shape[1]-predtill, data.shape[1])]
			except KeyboardInterrupt:
				raise
			except:
				pass
			try:
				params, _ = curve_fit(cyclic, range(len(weeks)), trend, method='trf', p0=(np.min(trend)-np.max(trend), 2, 1/8, 0, -np.max(trend)), bounds=self.cycdsbounds, sigma=conf)
				m1, k, f, o, b1 = params[0], params[1], params[2], params[3], params[4]
				err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp, m1, k, f, o, b1) for tmp in range(len(weeks))])-trend)))
				errs.append(err1)
				fits.append(params)
				if err1 < errf and err1 < self.explain_factor*err_l:
					errc = err1
					pred_tmp = [cyclic(tmp, m1, k, f, o, b1) for tmp in range(data.shape[1]-predtill, data.shape[1])]
			except KeyboardInterrupt:
				raise
			except:
				pass
			try:
				params, _ = curve_fit(cyclic, range(len(weeks)), trend, method='trf', p0=(np.min(trend)-np.max(trend), 4, 1/8, 0, -np.max(trend)), bounds=self.cycdsbounds, sigma=conf)
				m1, k, f, o, b1 = params[0], params[1], params[2], params[3], params[4]
				err1 = np.sqrt(np.sum(np.square(np.array([cyclic(tmp, m1, k, f, o, b1) for tmp in range(len(weeks))])-trend)))
				errs.append(err1)
				fits.append(params)
				if err1 < errf and err1 < self.explain_factor*err_l:
					pred_tmp = [cyclic(tmp, m1, k, f, o, b1) for tmp in range(data.shape[1]-predtill, data.shape[1])]
			except KeyboardInterrupt:
				raise
			except:
				pass


			if i%100 == 0:
				print("Done", i, "/", data.shape[2])
			pred.append(pred_tmp)
		pred = np.expand_dims(np.array(pred).T, axis=0)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
