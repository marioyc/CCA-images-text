# CCA-images-text
Canonical Correlation Analysis for joint representations of images and text based on [1].

Finds a common representation space for images and tags. Uses the [MS COCO dataset](http://mscoco.org/dataset/#overview), in particularly the captions given for the training data.

* main_count_and_features.py : computes tags for the training images and computes the features using the VGG16 network.
* main_cca.py : computes a PCA on the training data and then performs a CCA to find the projection matrices
* main_test.py : finds the corresponding tags for the images on the validation data

[1] Yunchao Gong, Qifa Ke, Michael Isard, Svetlana Lazebnik. A Multi-View Embedding Space for Modeling Internet Images, Tags, and their Semantics. International Journal of Computer Vision, Volume 106 Issue 2, January 2014, Pages 210-233.
