import pickle
import csv
from os.path import isfile
import numpy as np
from models.geostyle import GeoStyle
from event_model.outlier import OutlierDetector
from event_model.grouping import Grouper

gs = GeoStyle()
od = OutlierDetector()
grouper = Grouper()

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

input_file = 'metadata.pkl'
if not isfile(input_file):
	print("FileNotFound: Download 'metadata.pkl' and place it in the working directory")
	exit()
# This file contains all the saved events
output_files = 'events.csv'


data = unpickle(input_file)
cities = sorted(data['cities'].keys())
attributes = data['attributes']
categories = data['categories']

trends = []
confs = []
tots = []

first_iter = True

ids = []
for i in range(len(attributes)):
	for j in range(len(categories[i])):
		for cind, city in enumerate(cities):
			ids.append((attributes[i], categories[i][j], data['cities'][city]))
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
			tot = []
			for k in range(len(pos_tot)):
				if pos_tot[k][0] == 0:
					pos_tot[k][0] = 1
				elif pos_tot[k][0] == pos_tot[k][1]:
					pos_tot[k][0] = pos_tot[k][0]-1
				trend.append(pos_tot[k][0]/pos_tot[k][1])
				# 95% Binomial confidence interval
				conf.append(1.96*np.sqrt(pos_tot[k][0]*(pos_tot[k][1]-pos_tot[k][0])/((pos_tot[k][1])**3)))
				tot.append(pos_tot[k][1])
			trend = np.array(trend)
			conf = np.array(conf)
			tot = np.array(tot)
			trends.append(trend)
			confs.append(conf)
			tots.append(tot)

trends = np.array(trends)
confs = np.array(confs)
tots = np.array(tots)

data = np.expand_dims(trends.T, axis=0)
confs = np.expand_dims(confs.T, axis=0)
tots = np.expand_dims(tots.T, axis=0)


all_events = []
all_events_logsaliency = []
for i in range(data.shape[2]):
	trend = data[:, :, i:i+1]
	conf = confs[:, :, i:i+1]
	tot = tots[:, :, i:i+1]
	pred = gs.curve_fit(trend, conf)
	outliers, inds, scores = od.get_outliers(trend[0, :, 0], conf[0, :, 0], pred[0, :, 0], tot[0, :, 0])
	partitions, costs = grouper.get_best_partition(inds, scores)

	if partitions is not None:
		saliencys = 1/np.array(costs)
		for j in range(len(partitions)):
			all_events.append((ids[i], [weeks[tmp] for tmp in partitions[j]]))
			# log20 because the for a single outlier during hyp. testing score was cut-off at 0.05 = 1/20
			all_events_logsaliency.append(np.log(saliencys[j])/np.log(20))
	if i%100 == 99:
		print("Done", i+1, "/", data.shape[2])

# sort events by saliency
sorted_args = np.argsort(all_events_logsaliency)[::-1]

with open(output_files, 'w') as ofd:
	writer = csv.writer(ofd, delimiter=',')
	for ind in sorted_args:
		writer.writerow([all_events[ind][0][0], all_events[ind][0][1], all_events[ind][0][2],\
		all_events[ind][1], all_events_logsaliency[ind]])
