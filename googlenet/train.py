import os
import argparse
import numpy as np
from scipy.misc import imread, imresize, imshow

from numpy import random
from os.path import join
from googlenet import GoogleNet
from utils.manifest_api import Annotations
from utils.utils import get_filename_from_url
from utils.logger import Logger

################################################################################
# Constants
image_dim = (224, 224, 3)
random.seed(42)
################################################################################
#Argument parsing
parser = argparse.ArgumentParser(description='Arguments for training')
parser.add_argument('--mode', '-m', required=True)
parser.add_argument('--data-dir', '-dd', required=True)
parser.add_argument('--model-name', '-mn', required=True)
parser.add_argument('--learning-rate', '-lr', type=float,default=0.0001)
parser.add_argument('--batch-size', '-bs', type=int, default=32)
parser.add_argument('--iterations', '-i', type=int, default=6000)
parser.add_argument('--display-step', '-ds', type=int, default=100)

args = parser.parse_args()

trainOrTest = args.mode
input_dir = args.data_dir
model_name = args.model_name
learning_rate = args.learning_rate
batch_size = args.batch_size
pt_iters = args.iterations
display_step = args.display_step
#################################################################################
#Manifest files
annotations = Annotations("data/streetstyle27ktrain.manifest")
attributes = annotations.get_attributes()[-12:]
categories = []
for attribute in attributes:
	categories.append(annotations.get_categories(attribute))
#################################################################################
# Init network
googlenet = GoogleNet(0.5, learning_rate, join('models', model_name), attributes, categories, image_dim[0])
################################################################################
if trainOrTest not in ["test", "train", "continue"]:
	print("argument can be one of 'test','train','continue'")
	exit()

if trainOrTest in ["train", "continue"]:
	logger = Logger("GoogleNet_finetuning_inmemory_"+model_name)
else:
	logger = Logger("GoogleNet_testing_inmemory_"+model_name)
################################################################################

aug_randgen = random.RandomState(42)

# A function to dynamically keep images in memory when loaded for first time
# Since there are just 27K images, can be done in-memory
image_dataset = {}
def read_image(fname, crop):
	if fname not in image_dataset:
		image_dataset[fname] = imresize(imread(get_filename_from_url(fname, input_dir))[crop[1]:crop[3], crop[0]:crop[2]], image_dim[:2])
	return image_dataset[fname]

def augment_image(image):
	if aug_randgen.uniform() < 0.5:
		return image[:, ::-1, :]
	return image

def get_training_batches(index, batch_size=32):
	
	data = []
	data_x = np.empty((batch_size,)+image_dim)
	data_y = np.zeros((batch_size, len(categories[index])))

	for category in categories[index]:
		data.append(annotations.get_category_files(attributes[index], category))

	for i in range(batch_size):
		randcat = random.randint(0, len(categories[index]))
		randimg = random.randint(0, len(data[randcat]))
		data_x[i] = augment_image(read_image(data[randcat][randimg][0], data[randcat][randimg][1:5]))
		data_y[i][randcat] = 1
	return data_x, data_y


def get_accuracy(index, batch_size=32, validation=True):
	if validation:
		anno = Annotations("data/streetstyle27kvalidation.manifest")
	else:
		anno = Annotations("data/streetstyle27ktest.manifest")

	data = []
	data_all = []
	length = []

	for category in categories[index]:
		data.append(anno.get_category_files(attributes[index], category))
		data_all += data[-1]
		length.append(len(data[-1]))

	logger.logger.info("total data: %d", len(data_all))
	data_y = np.zeros((len(data_all),len(categories[index])))

	total = 0
	for j in range(len(length)):
		l = length[j]
		for i in range(l):
			data_y[total, j] = 1
			total += 1

	data_x = np.zeros((batch_size,)+image_dim)

	Acc = []

	for i in range(len(data_all)//batch_size-1):
		logger.logger.info("Tested: %d/%d", i, len(data_all)//batch_size-1)
		for j in range(i*batch_size, (i+1)*batch_size):
			fname = get_filename_from_url(data_all[j][0], input_dir)
			img = imread(fname)[data_all[j][2]:data_all[j][4], data_all[j][1]:data_all[j][3]]
			img = imresize(img, image_dim[:2])
			data_x[j%batch_size] = img
		Acc.append(googlenet.get_accuracy(data_x, data_y[i*batch_size:(i+1)*batch_size], attributes[index]))
	return np.mean(Acc)


def get_confusion(index, batch_size=32, validation=True):
	if validation:
		anno = Annotations("data/streetstyle27kvalidation.manifest")
	else:
		anno = Annotations("data/streetstyle27ktest.manifest")

	data = []
	data_all = []
	length = []

	for category in categories[index]:
		data.append(anno.get_category_files(attributes[index], category))
		data_all += data[-1]
		length.append(len(data[-1]))

	logger.logger.info("total data: %d", len(data_all))
	confusion = np.zeros((len(categories[index]), len(categories[index])))
	data_y = np.zeros((len(data_all), len(categories[index])))

	total = 0
	for j in range(len(length)):
		l = length[j]
		for i in range(l):
			data_y[total, j] = 1
			total += 1

	data_x = np.zeros((batch_size,)+image_dim)

	for i in range(len(data_all)//batch_size+1):
		if (i+1)%10 == 0:
			logger.logger.info("Tested: %d/%d", i+1, len(data_all)//batch_size+1)
		for j in range(i*batch_size, min((i+1)*batch_size, len(data_all))):
			fname = get_filename_from_url(data_all[j][0], input_dir)
			img = imread(fname)[data_all[j][2]:data_all[j][4], data_all[j][1]:data_all[j][3]]
			img = imresize(img, image_dim[:2])
			data_x[j%batch_size] = img

		pred_class = googlenet.get_class(data_x, attributes[index])
		true_class = np.argmax(data_y[i*batch_size:min((i+1)*batch_size, len(data_all))], axis=1)
		for j in range(true_class.shape[0]):
			confusion[true_class[j], pred_class[j]] += 1
	print(confusion)


	acc = np.trace(confusion)/np.sum(confusion)
	confusion = (confusion/np.sum(confusion, axis=1, keepdims=True))
	# print(confusion)
	macc = np.trace(confusion)/confusion.shape[0]
	return confusion, (acc, macc)


if trainOrTest in ["train", "continue"]:
	if trainOrTest == "continue":
		googlenet.restore_model()
	logger.logger.info("Finetuning Begins")
	step = 1
	while step <= pt_iters:
		batch_xs, batch_ys = [], []
		for a_ind in range(len(attributes)):
			batch_x, batch_y = get_training_batches(a_ind, batch_size = batch_size)
			batch_xs.append(batch_x)
			batch_ys.append(batch_y)
		googlenet.run_training(batch_xs, batch_ys)
		for a_ind in range(len(attributes)):
			if step%10 == 0:
				logger.logger.info("Iteration: %d,learning rate %f, training set accuracy on %s: %f and training loss: %f",\
					step, googlenet.get_lr(), attributes[a_ind], googlenet.get_accuracy(batch_xs[a_ind], batch_ys[a_ind], attributes[a_ind]), googlenet.get_loss(batch_xs[a_ind], batch_ys[a_ind], attributes[a_ind]))

		if step % display_step == 0:
			googlenet.save_model()
			for a_ind in range(len(attributes)):
				conf, accs = get_confusion(a_ind, batch_size = batch_size)
				logger.logger.info("Validation Accuracy on after %d iterations on %s: %f", step, attributes[a_ind], accs[0])
				logger.logger.info("Validation Mean class accuracy on after %d iterations on %s: %f", step, attributes[a_ind], accs[1])
				logger.logger.info(conf)
		step += 1
	googlenet.save_model()
elif trainOrTest == "test":
	Confusion = {}
	googlenet.restore_model()
	for i in range(len(attributes)):
		train = False
		logger.logger.info("Confusion on %s", attributes[i])
		Confusion[attributes[i]], accuracies = get_confusion(i, validation=False, batch_size = batch_size)
		logger.logger.info(Confusion[attributes[i]])
		logger.logger.info("Accuracy on %s : %f", attributes[i], accuracies[0])
		logger.logger.info("Mean class accuracy on %s : %f", attributes[i], accuracies[1])
