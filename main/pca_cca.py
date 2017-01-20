from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.externals import joblib
import argparse
import logging
import numpy as np
import os
import time

import cca

logging.basicConfig(filename='cca.log', format='%(asctime)s %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--numCC', default=15, type=int, help='number of components')
parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
parser.add_argument('--npca', default=-1, type=int, help='number of points used to calculate PCA')
args = parser.parse_args()

assert os.path.isfile('train_features.npz')
logging.info('Loading features file')
train_features = np.load('train_features.npz')
img_features = train_features['img_features']
tag_features = train_features['tag_features']

N_PCA = img_features.shape[0] if args.npca == -1 else args.npca
logging.info('Training: PCA of image features, N_PCA = %d', N_PCA)
start = time.time()
pca = IncrementalPCA(n_components=500, batch_size=512)
pca.fit(img_features[:N_PCA,:])
end = time.time()
logging.info('Time: %.4fm', (end - start) / 60)

logging.info('Apply PCA to image features')
start = time.time()
X = pca.transform(img_features)
end = time.time()
logging.info('Time: %.4fm', (end - start) / 60)

logging.info('Training: fit CCA')
start = time.time()
W_img, W_tag = cca.fit(X, tag_features, numCC=args.numCC, useGPU=args.gpu)
end = time.time()
logging.info('Time: %.4fm', (end - start) / 60)

np.savez('projections', pca=pca, W_img=W_img, W_tag=W_tag)
