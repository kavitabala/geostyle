from os.path import join
import numpy as np
from imageio import imread
from googlenet_infer import GoogleNet
from utils.manifest_api import Annotations
from PIL import Image
import requests


def download_image(id):
	url = f"https://i.vsco.co/{id}?w=224"
	response = requests.get(url)
	try:
		image = Image.open(BytesIO(response.content))
		return image
	except Exception as e:
		print(e)
		print(f'unable to get image data for url: {url}')
		return [], ''


input_dir = 'example_images'
images = ['517e7840514601336ff8dc2f8e33db3d_916651760287028794_296662372.jpg', \
	'e696d563a25709ba3434099b0815b2c8_816281954418255285_808992648.jpg', 'lilnasx.jpg', 'ladygaga.jpg']

# x1 y1 x2 y2
crops = [[140, 205, 225, 375], [297, 201, 385, 376]]

#################################################################################
attributes = ['clothing_pattern', 'major_color', 'wearing_necktie', 'collar_presence', 'wearing_scarf',\
	'sleeve_length', 'neckline_shape', 'clothing_category', 'wearing_jacket', 'wearing_hat', 'wearing_glasses', 'multiple_layers']
categories = [['Solid', 'Striped', 'Graphics', 'Floral', 'Plaid', 'Spotted'], \
	['Black', 'Red', 'White', 'Blue', 'Gray', 'Yellow', 'More than 1 color', 'Brown', 'Green', 'Pink', 'Orange', 'Purple', 'Cyan'], \
	['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], ['Short sleeve', 'Long sleeve', 'No sleeve'], ['Round', 'Folded', 'V-shape'], \
	['Dress', 'Outerwear', 'T-shirt', 'Suit', 'Shirt', 'Sweater', 'Tank top'], ['No', 'Yes'], ['No', 'Yes'], ['No', 'Yes'], \
	['One layer', 'Multiple layers']]

#################################################################################
glnet = GoogleNet(attributes, categories, 224)

# read
images = [imread(join(input_dir, fname)) for fname in images]
# crop
#images = [images[i][crops[i][1]:crops[i][3], crops[i][0]:crops[i][2]] for i in range(len(images))]

size = (224,224)
images = [Image.fromarray(image).resize(size) for image in images]
#images = [imresize(image, (224, 224)) for image in images]
images = np.array([np.asarray(i) for i in images])
print(images)

classes = glnet.get_classes(images)
for image in classes:
	for i in range(len(image)):
		print(attributes[i],': ',categories[i][image[i]])
	print()

