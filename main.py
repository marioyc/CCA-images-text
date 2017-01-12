from gensim.models import word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
import logging
import progressbar
import nltk
import numpy as np
import time

import cca

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

annFile = 'annotations/captions_train2014.json'
coco_train = COCO(annFile)
ids = coco_train.getAnnIds()
annotations = coco_train.loadAnns(ids)

cont = {}
logging.info('Count word frequencies, number of annotations = %d', len(annotations))
img_words = {}
bar = progressbar.ProgressBar()
for ann in bar(annotations):
    caption = ann['caption']
    image_id = ann['image_id']

    if image_id not in img_words:
        img_words[image_id] = set()

    for w in nltk.word_tokenize(caption):
        w = w.lower()
        img_words[image_id].add(w)
        if w in cont:
            cont[w] += 1
        else:
            cont[w] = 1
logging.info('Training: number of images = %d', len(img_words))

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
net = VGG16(weights='imagenet', include_top=False)
img_features = np.zeros((len(img_words), 512 * 7 * 7), dtype=np.float32)
tag_features = np.zeros((len(img_words), 200), dtype=np.float32)
pos = 0
counter_not_in_vocab = 0
logging.info('Training: calculate image features, choose tag for each image')
bar = progressbar.ProgressBar()
for image_id, words in bar(img_words.iteritems()):
    file_name = coco_train.imgs[image_id]['file_name']
    img = image.load_img('train2014/' + file_name, target_size=(224, 224))

    tag, best = '', -1
    for w in words:
        if w in model.wv.vocab and (best == -1 or cont.get(w,0) < best):
            tag, best = w, cont.get(w,0)

    if best != -1:
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = features.reshape(-1)
        img_features[pos,:] = features
        tag_features[pos,:] = model[tag]
    else:
        counter_not_in_vocab += 1

    pos += 1
    if pos % 5000 == 0:
        logging.info('Training: saving features calculated for the first %d images', pos)
        np.save('img_features_train', img_features[:pos,:])
        np.save('tag_features_train', tag_features[:pos,:])

logging.info('Training: saving features calculated for all the images')
np.save('img_features_train', img_features)
np.save('tag_features_train', tag_features)

assert counter_not_in_vocab == 0

logging.info('Training: fit CCA')
start = time.time()
W_img, W_tag = cca.cca(img_features, tag_features, numCC=15)
np.save('W_img', W_img)
np.save('W_tag', W_tag)
end = time.time()
logging.info('Time: %.4fm', (end - start) / 60)

annFile = 'annotations/captions_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

img_list = []
img_ids = []
logging.info('Testing: loading images and calculate image embeddings')
for ann in annotations[:10]:
    file_name = coco_val.imgs[ ann['image_id'] ]['file_name']
    img = image.load_img('val2014/' + file_name, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = net.predict(img)
    features = features.reshape(img.shape[0], - 1)
    img_list.append(features)
    img_ids.append(ann['image_id'])

img_features = np.array(img_list)

tag_keys = []
tag_list = []
logging.info('Testing: get embedding of all words in the vocabulary')
for k in model.wv.vocab.keys():
    tag_keys.append(k)
    tag_list.append(model[k])

logging.info('Testing: prediction')
N_RESULTS = 10
pos = 0
bar = progressbar.ProgressBar()
for img in bar(img_features):
   v_img = np.dot(img, W_img)
   scores = []
   for tag in tag_list:
       v_tag = np.dot(tag, W_tag)
       scores.append(1 - distance.cosine(v_img, v_tag))
   print scores[:N_RESULTS]
   index = np.argsort(scores)[::-1]
   print coco_val.imgs[ img_ids[pos] ]
   for i in range(N_RESULTS):
       print tag_keys[ index[i] ]
   pos += 1
