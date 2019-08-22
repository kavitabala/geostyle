import numpy as np
from statsmodels.tsa.ar_model import AR

class AutoRegression:
	def __init__(self):
		pass

	def predict(self, data, gap=0, predtill=1):
		assert predtill-1 <= gap
		true = data[:, -predtill:, :]
		pred = []
		for i in range(data.shape[2]):
			arm = AR(data[0, :-gap-1, i])
			fitted = arm.fit()
			pred.append(fitted.predict(start=data.shape[1]-gap-1, end=data.shape[1]-1)[gap-predtill:gap])
		pred = np.expand_dims(np.array(pred).T, axis=0)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
