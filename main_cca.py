from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.externals import joblib
import argparse
import numpy as np
import os
import time

import cca

parser = argparse.ArgumentParser()
parser.add_argument('--numCC', default=15, type=int, help='number of components')
parser.add_argument('--gpu', action='store_true', default=False, help='use gpu')
args = parser.parse_args()

assert os.path.isfile('img_features_train.npy')
print 'Loading image features file'
img_features = np.load('img_features_train.npy')

assert os.path.isfile('tag_features_train.npy')
print 'Loading tag features file'
tag_features = np.load('tag_features_train.npy')

print 'Training: PCA of image features'
start = time.time()
pca = IncrementalPCA(n_components=500, batch_size=32)
pca.fit_transform(img_features)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)
joblib.dump(pca, 'pca_img.pkl')

print 'Training: fit CCA'
start = time.time()
W_img, W_tag = cca.fit(img_features, tag_features, numCC=args.numCC, useGPU=args.gpu)
np.save('W_img', W_img)
np.save('W_tag', W_tag)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)
