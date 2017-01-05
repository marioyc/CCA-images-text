from gensim.models import word2vec
from keras.preprocessing import image
from pycocotools.coco import COCO
from sklearn.cross_decomposition import CCA
import nltk
import numpy as np
import time

import image_features

annFile = 'annotations/captions_train2014.json'
coco = COCO(annFile)
ids = coco.getAnnIds()
annotations = coco.loadAnns(ids)

cont = {}
print "Counting word frequencies"
for ann in annotations:
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
print "Loading images, choose tags"
for ann in annotations[:100]:
    file_name = coco.imgs[ ann['image_id'] ]['file_name']
    img = image.load_img('train2014/' + file_name, target_size=(224, 224))

    caption = ann['caption']
    tag, best = '', -1
    for w in nltk.word_tokenize(caption):
        w = w.lower()
        if w in model.vocab and (best == -1 or cont[w] < best):
            tag, best = w, cont[w]

    if best != -1:
        img_list.append(image.img_to_array(img))
        tag_list.append(model[tag])
    else:
        counter_not_in_vocab += 1

print "Caption with no word in vocab:", counter_not_in_vocab

print "Calculate image embeddings"
start = time.time()
img_features = image_features.vgg16_features(np.array(img_list))
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)

print "fit CCA"
start = time.time()
cca = CCA(n_components=1)
cca.fit(img_features, tag_list)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)
