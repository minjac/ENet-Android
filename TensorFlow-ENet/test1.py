import tensorflow as tf
import cv2
#from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
#slim = tf.contrib.slim

img_dir = 'D:\\TensorFlow-ENet-master\\training\\test_img'
photo_dir = 'D:\\TensorFlow-ENet-master\\training\\test_img\\save'
label_to_colours =    {0: [128,128,128],
                     1: [128,0,0],
                     2: [192,192,128],
                     3: [128,64,128],
                     4: [60,40,222],
                     5: [128,128,0],
                     6: [192,128,128],
                     7: [64,64,128],
                     8: [64,0,128],
                     9: [64,64,0],
                     10: [0,128,192],
                     11: [0,0,0]}

def grayscale_to_colour(image):
    print('Converting image...')
    image = image.reshape((360, 480, 1))
    image = np.repeat(image, 3, axis=-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(label_to_colours[label])
    return image

for i in range(1):
    print("i = %d"% i )
    img_path = img_dir + '/img'+ str( i + 1 ) + '.png'
    print(img_path)
    img = cv2.imread(img_path)
    img = np.array(img)
    print('img type:' + str(type(img)))
    print('img shape:' + str(np.shape(img)))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    if i == 0:
        images = img
    else:
        images = np.vstack((images,img))
print(np.shape(images))
#img = load_img(img_path)  # 输入预测图片的url
#img = img_to_array(img)
#img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的

# 加载模型
sess = tf.Session()
with open("D:\\TensorFlow-ENet-master\\checkpoint\\enet_test2.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    output = tf.import_graph_def(graph_def, input_map={"batch_1:0": images},#/fifo_queue
                                 return_elements=["Reshape_2:0"])#Images/Validation_segmentation_output
    # input_map 就是指明 输入是什么；
    # return_elements 就是指明输出是什么；两者在前面已介绍

result = sess.run(output)
#str=result.decode('utf-8')
print(result)
#print(result[0].shape)

segmentations = result
print(np.shape(segmentations))
# print segmentations.shape
# Stop at the 233rd image as it's repeated
print(type(segmentations[0]))
print(np.shape(segmentations))
test_img = segmentations[0][0]
test_img = test_img.astype(np.int32)
print(np.shape(test_img))
converted_image = grayscale_to_colour(test_img)
#plt.imshow(converted_image)
imsave(photo_dir + "/image1.png", converted_image)