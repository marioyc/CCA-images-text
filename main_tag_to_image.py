from gensim.models import word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
from scipy.spatial import distance
from sklearn.externals import joblib
import logging
import numpy as np
import os
import pickle
import time

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

def calc_testing_image_features():
    N_TEST = len(img_info)
    logging.info('Testing: number of image = %d', N_TEST)

    net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    assert os.path.isfile('pca_img.pkl')
    pca = joblib.load('pca_img.pkl')

    assert os.path.isfile('W_img.npy')
    W_img = np.load('W_img.npy')

    img_features = np.zeros((N_TEST, W_img.shape[1]), dtype=np.float32)

    pos = 0
    logging.info('Testing: precalculate image features')
    for image_id, info in img_info.iteritems():
        file_name = info['file_name']
        img = image.load_img('val2014/' + file_name, target_size=(224, 224))

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = pca.transform(features)
        features = np.dot(features, W_img)
        img_features[pos,:] = features

        pos += 1

    np.save('img_features_testing', img_features)
    return img_features

annFile = 'annotations/captions_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

img_info = {}
logging.info('Testing: get all different image ids')
for ann in annotations:
    image_id = ann['image_id']
    img_info[image_id] = coco_val.imgs[image_id]

img_ids = []
for image_id, info in img_info.iteritems():
    img_ids.append(image_id)

if not os.path.isfile('img_features_testing.npy'):
    img_features = calc_testing_image_features()
else:
    img_features = np.load('img_features_testing.npy')

N_RESULTS = 10
tags = ['cat', 'desktop', 'kitchen', 'group', 'beach', 'food', 'building', 'tower', 'book', 'computer', 'television']

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)

assert os.path.isfile('W_tag.npy')
W_tag = np.load('W_tag.npy')

f = open('test_t2i.txt', 'w')
for tag in tags:
    f.write(tag + '\n')
    features = model[tag]
    features = np.dot(features, W_tag)

    scores = np.zeros(len(img_info))
    for i in range(len(img_info)):
        scores[i] = distance.euclidean(img_features[i,:], features)

    index = np.argsort(scores)
    for i in range(N_RESULTS):
        ind = index[i]
        f.write(img_info[ img_ids[ind] ]['flickr_url'] + '\n')
