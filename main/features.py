from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np

def calc_testing_image_features(img_info, pca, W_img):
    N_TEST = len(img_info)
    logging.info('Testing: number of image = %d', N_TEST)

    net = VGG16(weights='imagenet', include_top=True)
    net.layers.pop()
    net.outputs = [net.layers[-1].output]
    net.layers[-1].outbound_nodes = []

    img_features = np.zeros((N_TEST, W_img.shape[1]), dtype=np.float32)
    img_ids = []

    pos = 0
    logging.info('Testing: precalculate image features')
    for image_id, info in img_info.iteritems():
        file_name = info['file_name']
        img = image.load_img('val2014/' + file_name, target_size=(224, 224))
        img_ids.append(image_id)

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = net.predict(img)
        features = pca.transform(features)
        features = np.dot(features, W_img)
        img_features[pos,:] = features

        pos += 1

    np.savez_compressed('test_features', img_ids=img_ids, img_features=img_features)
    return img_ids, img_features
