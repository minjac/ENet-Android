import tensorflow as tf
#from create_tf_record import *
import os
from tensorflow.python.framework import graph_util
import cv2
import numpy as np
from preprocessing import preprocess

os.environ['CUDA_VISIBLE_DEVICES']='2'
slim = tf.contrib.slim

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', 'D:\\TensorFlow-ENet-master\\dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/original', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')

#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 25, 'The batch size used for validation.')
#flags.DEFINE_integer('image_height', 360, "The inp ut height of the images.")
#flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 100, "The number of epochs to train your model.")
#flags.DEFINE_integer('num_epochs', 100, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 4e-4, 'The initial learning rate for your training.')
#flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size #Can be larger than train_batch as no need to backpropagate gradients.
combine_dataset = FLAGS.combine_dataset
#Visualization and where to save images
save_images = FLAGS.save_images
photo_dir = os.path.join(FLAGS.logdir, "images")

#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "\\val") if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "\\valannot") if file.endswith('.png')])


images_val = tf.convert_to_tensor(image_val_files)
annotations_val = tf.convert_to_tensor(annotation_val_files)
input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])

# Decode the image and annotation raw content
image_val = tf.read_file(input_queue_val[0])
image_val = tf.image.decode_jpeg(image_val, channels=3)
annotation_val = tf.read_file(input_queue_val[1])
annotation_val = tf.image.decode_png(annotation_val)

preprocessed_image_val, preprocessed_annotation_val = preprocess(image_val, annotation_val, image_height, image_width)
images_val, annotations_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val], batch_size=eval_batch_size, allow_smaller_final_batch=True)
print(tf.shape(images_val))
model_path = "D:\\TensorFlow-ENet-master\\checkpoint\\enet_test.pb"


def freeze_graph_test(pb_path, image_path):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(pb_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                # 定义输入的张量名称,对应网络结构的输入张量
                input_image_tensor = sess.graph.get_tensor_by_name("batch_1:0")
                #softmax = sess.graph.get_tensor_by_name("softmax/softmax:0")
                conv62 = sess.graph.get_tensor_by_name("Images/Validation_segmentation_output:0")
                # 定义输出的张量名称

                img = cv2.imread(image_path)
                img = (img - 127.5) * 0.0078125
                img_x = np.expand_dims(img, 0)
                img_x.astype(np.float32)
                # 读取测试图片
#                im=read_image(image_path,resize_height,resize_width,normalization=True)
#                im=im[np.newaxis,:]
# 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
                out1 = sess.run(conv62, feed_dict={input_image_tensor: img_x})#out1, out2 = sess.run([softmax, conv62], feed_dict = {input_image_tensor:img_x})
                out_conv61 = np.array(out1)
                #out_conv62 = np.array(out2)
                print("---------------")
                print(out_conv61)
                print("---------------")
                #print(out_conv62)
                print("test done")

if __name__ == '__main__':
    freeze_graph_test(model_path, images_val)