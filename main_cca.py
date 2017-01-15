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
parser.add_argument('--npca', default=-1, type=int, help='number of points used to calculate PCA')
parser.add_argument('--pcaFile', default='pca_img', type=str)
parser.add_argument('--imageMatrix', default='W_img', type=str)
parser.add_argument('--tagMatrix', default='W_tag', type=str)
args = parser.parse_args()

assert os.path.isfile('img_features_train.npy')
print 'Loading image features file'
img_features = np.load('img_features_train.npy')

assert os.path.isfile('tag_features_train.npy')
print 'Loading tag features file'
tag_features = np.load('tag_features_train.npy')

N_PCA = img_features.shape[0] if args.pca == -1 else args.pca
print 'Training: PCA of image features, N_PCA = {}'.format(N_PCA)
start = time.time()
pca = IncrementalPCA(n_components=500, batch_size=512)
pca.fit(img_features[:N_PCA,:])
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)
joblib.dump(pca, args.pcaFile + '.pkl')

print 'Apply PCA to image features'
start = time.time()
X = pca.transform(img_features)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)

print 'Training: fit CCA'
start = time.time()
W_img, W_tag = cca.fit(X, tag_features, numCC=args.numCC, useGPU=args.gpu)
np.save(args.imageMatrix, W_img)
np.save(args.tagMatrix, W_tag)
end = time.time()
print 'Time: {0:.4f}m'.format((end - start) / 60)
