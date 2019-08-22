import numpy as np

class MeanPredictor:
	def __init__(self):
		pass

	def predict(self, data, gap=0, predtill=1):
		assert predtill-1 <= gap
		true = data[:, -predtill:, :]
		pred = np.repeat(np.mean(data[:, :-gap-1, :], axis=1, keepdims=True), predtill, axis=1)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
