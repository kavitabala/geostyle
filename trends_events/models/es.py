import numpy as np


class ExponentialSmoothing:
	def __init__(self):
		pass

	def exp_smooth(self, data, alpha):
		true = data[:, -1, :]
		pred = data[:, :-1, :]
		alphas = np.ones((1, pred.shape[1], 1))*alpha
		for i in range(2, pred.shape[1]+1):
			alphas[0, -i, 0] = (1-alpha)*alphas[0, -i+1, 0]
		alphas[0, 0, 0] = alphas[0, 0, 0]/alpha

		pred = np.sum(pred*alphas, axis=1)

		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred


	def predict(self, data, gap=0, predtill=1):
		assert predtill-1 <= gap
		malpha = 0
		mmae = 100
		for i in range(100):
			alpha = i*0.01+0.01
			mae, _, _ = self.exp_smooth(data[:, :-1-gap, :], alpha)
			if mae < mmae:
				mmae = mae
				malpha = alpha
		if gap != 0:
			tmp = data[:, :-gap, :]
		else:
			tmp = data[:, :, :]
		for i in range(gap+1):
			_, _, pred = self.exp_smooth(tmp, malpha)
			tmp = np.concatenate([tmp[:, :-1, :], np.expand_dims(pred, axis=0), np.expand_dims(data[:, i-gap, :], axis=1)], axis=1)
		true = data[:, -predtill:, :]
		pred = tmp[:, -predtill-1:-1, :]
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
