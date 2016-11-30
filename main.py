import numpy as np
from keras.preprocessing import image

import image_features

img = image.load_img('elephant1.jpg', target_size=(224, 224))
x = image.img_to_array(img)

img2 = image.load_img('elephant2.jpg', target_size=(224, 224))
y = image.img_to_array(img2)

z = np.array([x,y])

features = image_features.vgg16_features(z)
print features.shape
