import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#one_hot 为True表示输出的标签以one_hot编码方式进行编码，为False直接输出图片表示的数字
mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)

print("Training data size: ")
print(mnist.train.num_examples) #结果为55000, 说明训练集中有55000个样本

print("Validating data size: ")
print(mnist.validation.num_examples) ##结果为5000, 说明训练集中有55000个样本

print("Testing data size: ")
print(mnist.test.num_examples)

print(mnist.train.images[1])
plt.imshow(mnist.train.images[1].reshape(28,28), cmap='gray') #如果不加cmap='gray'会将灰度图显示为伪彩色图
plt.show()
print(mnist.train.labels[1])

