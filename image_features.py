from keras.applications.vgg16 import VGG16, preprocess_input

def vgg16_features(img_array):
    model = VGG16(weights='imagenet', include_top=False)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    features = features.reshape(img_array.shape[0], -1)
    return features
