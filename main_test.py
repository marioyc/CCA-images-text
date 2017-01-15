from gensim.models import word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
from scipy.spatial import distance
from sklearn.externals import joblib
import argparse
import logging
import numpy as np
import os
import pickle
import time

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--pcaFile', default='pca_img', type=str)
parser.add_argument('--imageMatrix', default='W_img', type=str)
parser.add_argument('--tagMatrix', default='W_tag', type=str)
parser.add_argument('--tagsFile', default='test_tags', type=str)
args = parser.parse_args()

annFile = 'annotations/captions_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
net = VGG16(weights='imagenet', include_top=True)
net.layers.pop()
net.outputs = [net.layers[-1].output]
net.layers[-1].outbound_nodes = []

assert os.path.isfile(args.pcaFile + '.pkl')
pca = joblib.load(args.pcaFile + '.pkl')

assert os.path.isfile(args.imageMatrix + '.npy')
W_img = np.load(args.imageMatrix + '.npy')

assert os.path.isfile(args.tagMatrix + '.npy')
W_tag = np.load(args.tagMatrix + '.npy')

assert os.path.isfile('possible_tags.pkl')
possible_tags = pickle.load(open('possible_tags.pkl', 'rb'))

tag_keys = []
tag_features_list = []
logging.info('Testing: get embedding of all possible tags')
for tag in possible_tags:
    tag_keys.append(tag)
    tag_features_list.append(model[tag])

img_info = {}
logging.info('Testing: get all different image ids')
for ann in annotations:
    image_id = ann['image_id']
    img_info[image_id] = coco_val.imgs[image_id]

N_TEST = len(img_info)
logging.info('Testing: number of images = %d', N_TEST)

N_RESULTS = 5
f = open(args.tagsFile + '.txt', 'w')
img_ids = []
pos = 0
logging.info('Testing: prediction')
for image_id, info in img_info.iteritems():
    file_name = info['file_name']
    img = image.load_img('val2014/' + file_name, target_size=(224, 224))

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    img_features = net.predict(img)
    img_features = img_features.reshape(img.shape[0], - 1)
    img_features = pca.transform(img_features)
    img_ids.append(image_id)

    v_img = np.dot(img_features, W_img)
    scores = np.zeros(len(tag_features_list))
    for i in range(len(tag_features_list)):
        tag_features = tag_features_list[i]
        v_tag = np.dot(tag_features, W_tag)
        scores[i] = distance.euclidean(v_img, v_tag)

    index = np.argsort(scores)
    f.write(coco_val.imgs[ img_ids[pos] ]['flickr_url'] + '\n')
    for i in range(N_RESULTS):
        f.write(tag_keys[ index[i] ] + '\n')

    pos += 1
