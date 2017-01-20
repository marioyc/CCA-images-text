from gensim.models import word2vec
from pycocotools.coco import COCO
from scipy.spatial import distance
import logging
import numpy as np
import os
import pickle
import time

import features

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

annFile = 'annotations/instances_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

assert os.path.isfile('projections.npz')
projections = np.load('projections.npz')
pca = projections['pca'].item()
W_img = projections['W_img']
W_tag = projections['W_tag']

assert os.path.isfile('possible_tags.pkl')
possible_tags = pickle.load(open('possible_tags.pkl', 'rb'))

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
tags = []
tag_features_list = []
logging.info('Testing: get embedding of all possible tags')
for tag in possible_tags:
    tags.append(tag)
    tag_features_list.append(model[tag])
N_TAGS = len(tags)

img_info = {}
logging.info('Testing: get all different image ids')
for ann in annotations:
    image_id = ann['image_id']
    if image_id not in img_info:
        img_info[image_id] = coco_val.imgs[image_id]

N_TEST = len(img_info)
logging.info('Testing: number of images = %d', N_TEST)

if not os.path.isfile('test_features.npz'):
    img_ids, img_features = features.calc_testing_image_features(img_info, pca, W_img)
else:
    test_features = np.load('test_features.npz')
    img_ids = test_features['img_ids']
    img_features = test_features['img_features']

W_tag = projections['W_tag']

N_RESULTS = 5
f = open('i2t_results.txt', 'w')
pos = 0
logging.info('Testing: prediction')
start = time.time()
for image_id in img_ids:
    v_img = img_features[pos]
    scores = np.zeros(N_TAGS)
    for i in range(N_TAGS):
        tag_features = tag_features_list[i]
        v_tag = np.dot(tag_features, W_tag)
        scores[i] = distance.euclidean(v_img, v_tag)

    index = np.argsort(scores)
    info = img_info[image_id]
    f.write(info['flickr_url'] + ' ' + info['coco_url'] + '\n')
    for i in range(N_RESULTS):
        ind = index[i]
        f.write(tags[ind] + ', ')
    f.write('\n')

    pos += 1
end = time.time()
logging.info('Time: %.4fm', (end - start) / 60)
