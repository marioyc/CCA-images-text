from gensim.models import word2vec
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from pycocotools.coco import COCO
import logging
import os
import pickle
import progressbar
import nltk
import numpy as np

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

annFile = 'annotations/captions_train2014.json'
coco_train = COCO(annFile)
ids = coco_train.getAnnIds()
annotations = coco_train.loadAnns(ids)

def count_words():
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

    pickle.dump(cont, open('cont.pkl', 'wb'))
    return cont

def calc_features():
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
        if pos % 10000 == 0:
            logging.info('Training: saving features calculated for the first %d images', pos)
            np.save('img_features_train', img_features[:pos,:])
            np.save('tag_features_train', tag_features[:pos,:])

    logging.info('Training: saving features calculated for all the images')
    np.save('img_features_train', img_features)
    np.save('tag_features_train', tag_features)

    assert counter_not_in_vocab == 0

    return img_features, tag_features

if os.path.isfile('cont.pkl'):
    cont = pickle.load(open('cont.pkl', 'rb'))
else:
    cont = count_words()

if not os.path.isfile('img_features_train.npy'):
    img_features, tag_features = calc_features()
