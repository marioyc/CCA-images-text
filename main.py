from keras.preprocessing import image
from pycocotools.coco import COCO
import numpy as np

import image_features

img = image.load_img('elephant1.jpg', target_size=(224, 224))
x = image.img_to_array(img)

img2 = image.load_img('elephant2.jpg', target_size=(224, 224))
y = image.img_to_array(img2)

z = np.array([x,y])

features = image_features.vgg16_features(z)
print features.shape

annFile = 'annotations/captions_train2014.json'
coco = COCO(annFile)
ids = coco.getAnnIds()
annotations = coco.loadAnns(ids)
cont = 0
for ann in annotations:
    caption = ann['caption']
    file_name = coco.imgs[ ann['image_id'] ]['file_name']
    if file_name[:10] == 'COCO_train':
        print file_name, caption
        cont += 1
        if cont == 5:
            break
