from gensim.models import word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
from scipy.spatial import distance
import logging
import os
import numpy as np
import time

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
net = VGG16(weights='imagenet', include_top=False)

assert os.path.isfile('W_img.npy')
W_img = np.load('W_img.npy')

assert os.path.isfile('W_tag.npy')
W_tag = np.load('W_tag.npy')

annFile = 'annotations/captions_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

img_info = {}
logging.info('Testing: get all different image ids')
for ann in annotations:
    image_id = ann['image_id']
    img_info[image_id] = coco_val.imgs[image_id]

img_features = np.zeros((10, W_img.shape[0]))
img_ids = []
pos = 0
logging.info('Testing: loading images and calculate image embeddings')
for image_id, info in img_info.iteritems():
    file_name = info['file_name']
    img = image.load_img('val2014/' + file_name, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = net.predict(img)
    features = features.reshape(img.shape[0], - 1)
    features = features[:,:5000]
    img_features[pos,:] = features
    img_ids.append(image_id)
    pos += 1
    if pos == 10:
        break

tag_keys = []
tag_list = []
logging.info('Testing: get embedding of all words in the vocabulary')
for k in model.wv.vocab.keys():
    tag_keys.append(k)
    tag_list.append(model[k])

logging.info('Testing: prediction')
N_RESULTS = 10
pos = 0
for img in img_features:
   v_img = np.dot(img, W_img)
   scores = []
   for tag in tag_list:
       v_tag = np.dot(tag, W_tag)
       scores.append(1 - distance.cosine(v_img, v_tag))
   #print scores[:N_RESULTS]
   index = np.argsort(scores)[::-1]
   print coco_val.imgs[ img_ids[pos] ]['flickr_url']
   results = []
   for i in range(N_RESULTS):
       results.append(tag_keys[ index[i] ])
   print results
   pos += 1
