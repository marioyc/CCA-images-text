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

    pca = projections['pca'].item()
    W_img = projections['W_img']

    img_features = np.zeros((N_TEST, W_img.shape[1]), dtype=np.float32)
    img_ids = []

    pos = 0
    logging.info('Testing: precalculate image features')
    for image_id, info in img_info.iteritems():
        file_name = info['file_name']
        img = image.load_img('val2014/' + file_name, target_size=(224, 224))
        img_ids.append(image_id)

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = pca.transform(features)
        features = np.dot(features, W_img)
        img_features[pos,:] = features

        pos += 1

    np.savez_compressed('test_features', img_ids=img_ids, img_features=img_features)
    return img_ids, img_features

annFile = 'annotations/instances_val2014.json'
coco_instances = COCO(annFile)
ids = coco_instances.getAnnIds()
annotations = coco_instances.loadAnns(ids)
categories = coco_instances.loadCats(coco_instances.getCatIds())

category_name = {}
for cat in categories:
    category_name[ cat['id'] ] = cat['name']

img_info = {}
img_categories = {}
logging.info('Testing: get info for all images')
for ann in annotations:
    image_id = ann['image_id']
    if image_id not in img_info:
        img_info[image_id] = coco_instances.imgs[image_id]
        img_categories[image_id] = set()
    category = category_name[ ann['category_id'] ]
    img_categories[image_id].add(category)

assert os.path.isfile('projections.npz')
projections = np.load('projections.npz')

if not os.path.isfile('test_features.npz'):
    img_ids, img_features = calc_testing_image_features()
else:
    test_features = np.load('test_features.npz')
    img_ids = test_features['img_ids']
    img_features = test_features['img_features']

N_IMGS = len(img_ids)
model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
W_tag = projections['W_tag']

assert os.path.isfile('possible_tags.pkl')
possible_tags = pickle.load(open('possible_tags.pkl', 'rb'))

N_RESULTS = 50
tags = [cat['name'] for cat in categories]

f = open('t2i_results.txt', 'w')
for tag in tags:
    if tag not in possible_tags:
        f.write(tag + ' is not in the list of possible tags\n')
        continue

    f.write('TAG: ' + tag + '\n')
    features = model[tag]
    features = np.dot(features, W_tag)

    scores = np.zeros(N_IMGS)
    for i in range(N_IMGS):
        scores[i] = distance.euclidean(img_features[i,:], features)

    index = np.argsort(scores)
    correct = 0
    for i in range(N_RESULTS):
        ind = index[i]
        image_id = img_ids[ind]
        info = img_info[image_id]
        #f.write(info['flickr_url'] + ' ' + info['coco_url'] + '\n')
        #for cat in img_categories[image_id]:
        #    f.write(cat + ', ')
        #f.write('\n')
        if tag in img_categories[image_id]:
            correct += 1
    f.write('Precision = {0:.2f}\n\n'.format(float(correct) / N_RESULTS))
