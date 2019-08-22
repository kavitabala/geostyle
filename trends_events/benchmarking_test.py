import numpy as np
from os.path import isfile
import pickle

# load models
from models.mean_pred import MeanPredictor
from models.last_pred import LastPredictor
from models.ar import AutoRegression
from models.var import VectorAutoRegression
from models.es import ExponentialSmoothing
from models.linear import Linear
from models.sine import Sinusoidal
from models.sinelin import SinusoidalLinear
from models.cyclic import Cyclic
from models.geostyle import GeoStyle
mean_pred = MeanPredictor()
last_pred = LastPredictor()
ar = AutoRegression()
var = VectorAutoRegression()
es = ExponentialSmoothing()
linear = Linear()
sine = Sinusoidal()
sinelin = SinusoidalLinear()
cyclic = Cyclic()
gs = GeoStyle()


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

input_file = 'metadata.pkl'
if not isfile(input_file):
	print("FileNotFound: Download 'metadata.pkl' and place it in the working directory")
	exit()
data = unpickle(input_file)
cities = sorted(data['cities'].keys())
attributes = data['attributes']
categories = data['categories']

trends_maincopy = []
confs_maincopy = []

first_iter = True
for i in range(len(attributes)):
	for j in range(len(categories[i])):
		for cind, city in enumerate(cities):
			pos_tot = []
			datum = data['classifications'][city]
				# remove weeks with small amount of data from start and end
			if first_iter:
				weeks = sorted(datum.keys())
				weeks = weeks[5:-5]
				first_iter = False
			for week in weeks:
				pos_tot.append([np.sum(datum[week][:, i] == j), datum[week].shape[0]])

			trend = []
			conf = []
			for k in range(len(pos_tot)):
				if pos_tot[k][0] == 0:
					pos_tot[k][0] = 1
				elif pos_tot[k][0] == pos_tot[k][1]:
					pos_tot[k][0] = pos_tot[k][0]-1
				trend.append(pos_tot[k][0]/pos_tot[k][1])
				# 95% Binomial confidence interval
				conf.append(1.96*np.sqrt(pos_tot[k][0]*(pos_tot[k][1]-pos_tot[k][0])/((pos_tot[k][1])**3)))
			trend = np.array(trend)
			conf = np.array(conf)
			trends_maincopy.append(trend)
			confs_maincopy.append(conf)

trends_maincopy = np.array(trends_maincopy)
confs_maincopy = np.array(confs_maincopy)

trends = np.copy(trends_maincopy[:, :-26])
confs = np.copy(confs_maincopy[:, :-26])
gap = 1
predtill = 1
data = np.expand_dims(trends.T, axis=0)
confs = np.expand_dims(confs.T, axis=0)

mae, mape, pred_mean = mean_pred.predict(data, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Naive mean:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = last_pred.predict(data, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Naive last:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = ar.predict(data, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Autoregression:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = var.predict(data, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Vector-autoregression:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = es.predict(data, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Exponential smoothing:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = linear.predict(data, confs, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Linear Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = sine.predict(data, confs, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Sinusoidal Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = sinelin.predict(data, confs, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Sinusoidal+Linear Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = cyclic.predict(data, confs, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) Cyclic Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = gs.predict(data, confs, gap=gap, predtill=predtill)
print("Next week (MAE MAPE) GeoStyle Model:", np.round(mae, 4), np.round(mape, 2))

trends = np.copy(trends_maincopy)
confs = np.copy(confs_maincopy)
gap = 26
predtill = 26
data = np.expand_dims(trends.T, axis=0)
confs = np.expand_dims(confs.T, axis=0)

mae, mape, pred_mean = mean_pred.predict(data, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Naive mean:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = last_pred.predict(data, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Naive last:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = ar.predict(data, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Autoregression:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = var.predict(data, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Vector-autoregression:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = es.predict(data, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Exponential smoothing:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = linear.predict(data, confs, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Linear Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = sine.predict(data, confs, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Sinusoidal Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = sinelin.predict(data, confs, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Sinusoidal+Linear Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = cyclic.predict(data, confs, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) Cyclic Model:", np.round(mae, 4), np.round(mape, 2))
mae, mape, pred_mean = gs.predict(data, confs, gap=gap, predtill=predtill)
print("Next 26 weeks (MAE MAPE) GeoStyle Model:", np.round(mae, 4), np.round(mape, 2))
