import tensorflow as tf

INPUT_NODE = 784 #图片大小为28*28展开之后就是一个784维的向量
OUTPUT_NODE = 10 #因为是one_hot编码，所以一个标签用一个10维向量表示，输出也用一个10维向量表示
LAYER1_NODE = 500

#如下get_variable 是一个变量管理函数
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

#如下是一个有：一个输入层，一个隐藏层的（隐藏层有500个神经元）的全连接网络
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2