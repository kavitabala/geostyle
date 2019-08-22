import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR

class VectorAutoRegression:
	def __init__(self):
		pass

	def predict(self, data, gap=0, predtill=1):
		assert predtill-1 <= gap
		true = data[:, -predtill:, :]
		pred = []
		varm = VAR(data[0, :-gap-1, :])
		fitted = varm.fit()
		pred = varm.predict(fitted.params, start=data.shape[1]-gap-2, end=data.shape[1]-1)[gap+1-predtill:gap+1]
		pred = np.expand_dims(np.array(pred), axis=0)
		mae = np.mean(np.abs(pred-true))
		mape = np.mean(np.abs(pred-true)/true)*100
		return mae, mape, pred
