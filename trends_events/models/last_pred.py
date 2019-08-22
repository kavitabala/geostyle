import numpy as np

class LastPredictor:
	def __init__(self):
		pass

	def predict(self, data, gap=0, predtill=1):
		assert predtill-1 <= gap
		true = data[:, -predtill:, :]
		pred = np.repeat(np.expand_dims(data[:, -2-gap, :], axis=1), predtill, axis=1)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
