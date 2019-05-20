import tensorflow as tf
import os

model_dir = '/mnt/mdisk/lwt/ljm/TensorFlow-ENet-master-original/log/original'#D:\\TensorFlow-ENet-master\\checkpoint#D:\\TensorFlow-ENet-master\\checkpoint
model_name = 'enet_test.pb'#'graph.pb'#enet_test2.pb

def create_graph():
    with tf.gfile.FastGFile(os.path.join(
            model_dir, model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')