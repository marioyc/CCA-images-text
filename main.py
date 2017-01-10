from gensim.models import word2vec
from keras.preprocessing import image
from pycocotools.coco import COCO
from scipy.spatial import distance
from sklearn.cross_decomposition import CCA
import nltk
import numpy as np
import progressbar
import time

import cca
import image_features

K_IMG = 500

annFile = 'annotations/captions_train2014.json'
coco_train = COCO(annFile)
ids = coco_train.getAnnIds()
annotations = coco_train.loadAnns(ids)
print len(annotations)

cont = {}
print "Counting word frequencies"
bar = progressbar.ProgressBar()
for ann in bar(annotations):
    caption = ann['caption']
    for w in nltk.word_tokenize(caption):
        w = w.lower()
        if w in cont:
            cont[w] += 1
        else:
            cont[w] = 1

model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
img_list = []
tag_list = []
counter_not_in_vocab = 0
print "Training: loading images, choose tag for each image"
bar = progressbar.ProgressBar()
for ann in bar(annotations[:8000]):
    file_name = coco_train.imgs[ ann['image_id'] ]['file_name']
    img = image.load_img('train2014/' + file_name, target_size=(224, 224))

    caption = ann['caption']
    tag, best = '', -1
    for w in nltk.word_tokenize(caption):
        w = w.lower()
        if w in model.vocab and (best == -1 or cont.get(w,0) < best):
            tag, best = w, cont.get(w,0)

    if best != -1:
        img_list.append(image.img_to_array(img))
        tag_list.append(model[tag])
    else:
        counter_not_in_vocab += 1
tag_features = np.array(tag_list)
del tag_list

print "Caption with no word in vocab:", counter_not_in_vocab

print "Training: calculate image embeddings"
start = time.time()
img_features = image_features.vgg16_features(np.array(img_list))
end = time.time()
del img_list
print 'Time: {0:.4f}m'.format((end - start) / 60)

print "Training: fit CCA"
start = time.time()
img_features = img_features[:,:K_IMG]
W_img, W_tag = cca.cca(img_features, tag_features, numCC=15)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)

annFile = 'annotations/captions_val2014.json'
coco_val = COCO(annFile)
ids = coco_val.getAnnIds()
annotations = coco_val.loadAnns(ids)

img_list = []
img_ids = []
print "Testing: loading images"
for ann in annotations[:10]:
    file_name = coco_val.imgs[ ann['image_id'] ]['file_name']
    img = image.load_img('val2014/' + file_name, target_size=(224, 224))
    img_list.append(image.img_to_array(img))
    img_ids.append(ann['image_id'])

print "Testing: calculate image embeddings"
start = time.time()
img_features = image_features.vgg16_features(np.array(img_list))
img_features = img_features[:,:K_IMG]
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)

tag_keys = []
tag_list = []
print "Testing: get embedding of all words in the vocabulary"
for k in model.vocab.keys():
    tag_keys.append(k)
    tag_list.append(model[k])

print "Testing: prediction"
N_RESULTS = 10
pos = 0
start = time.time()
print img_features.shape
for img in img_features:
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

end = time.time()
print 'Time : {0:.4f}m'.format((end - start) / 60)
