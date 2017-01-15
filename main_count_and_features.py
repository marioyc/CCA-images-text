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
img_count = {}
img_captions = {}

def count_words():
    stop = set(nltk.corpus.stopwords.words('english'))
    logging.info('Count word frequencies, number of annotations = %d', len(annotations))
    bar = progressbar.ProgressBar()
    for ann in bar(annotations):
        caption = ann['caption']
        image_id = ann['image_id']
        tokens = nltk.word_tokenize(caption)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if not w in stop]

        if image_id not in img_count:
            img_count[image_id] = {}
            img_captions[image_id] = [tokens]
        else:
            img_captions[image_id].append(tokens)

        for w in tokens:
            if w in img_count[image_id]:
                img_count[image_id][w] += 1
            else:
                img_count[image_id][w] = 1

    logging.info('Training: number of images = %d', len(img_count))

def calc_features():
    model = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)
    net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    TAGS_PER_IMAGE = 2
    img_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 4096), dtype=np.float32)
    tag_features = np.zeros((TAGS_PER_IMAGE * len(img_count), 200), dtype=np.float32)

    possible_tags = set()

    f = file('train_tags.txt', 'w')
    pos = 0
    logging.info('Training: calculate image features, choose tag for each image')
    bar = progressbar.ProgressBar()
    for image_id, words in bar(img_count.iteritems()):
        file_name = coco_train.imgs[image_id]['file_name']
        img = image.load_img('train2014/' + file_name, target_size=(224, 224))

        words_list = []
        words_count = []
        for w in words:
            if w in model.wv.vocab:
                words_list.append(w)
                words_count.append(img_count[image_id][w])

        words_count = np.array(words_count)
        index = np.argsort(words_count)[::-1]

        f.write(coco_train.imgs[image_id]['flickr_url'] + '\n')
        for i in range(TAGS_PER_IMAGE):
            f.write(words_list[ index[i] ] + '\n')
        #for i in range(0,min(5,len(index))):
        #    ind = index[i]
        #    print words_list[ind], words_count[ind]
        #for caption in img_captions[image_id]:
        #    print caption

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = features.reshape(-1)

        for i in range(TAGS_PER_IMAGE):
            ind = index[i]
            img_features[TAGS_PER_IMAGE * pos + i,:] = features
            tag_features[TAGS_PER_IMAGE * pos + i,:] = model[ words_list[ind] ]
            possible_tags.add(words_list[ind])

        pos += 1
        if pos % 10000 == 0:
            logging.info('Training: saving features calculated for the first %d images', pos)
            np.save('img_features_train', img_features[:pos,:])
            np.save('tag_features_train', tag_features[:pos,:])

    logging.info('Training: saving features calculated for all the images')
    np.save('img_features_train', img_features)
    np.save('tag_features_train', tag_features)

    logging.info('Training: number of possible tags = %d', len(possible_tags))
    pickle.dump(possible_tags, open('possible_tags.pkl', 'wb'))

count_words()

if not os.path.isfile('img_features_train.npy'):
    calc_features()
