import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
from PIL import Image
from scipy import ndimage
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from signsrecognition import initialize_parameters

## START CODE HERE ## (PUT YOUR IMAGE NAME)

my_image = "example1.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T

saver = tf.train.import_meta_graph("savedmodel/model.ckpt.meta")

#parameters = initialize_parameters()

sess = tf.Session()
saver.restore(sess, 'savedmodel/model.ckpt')
    #parameters = sess.run(parameters)
W1 = sess.run('W1:0')
b1 = sess.run('b1:0')
W2 = sess.run('W2:0')
b2 = sess.run('b2:0')
W3 = sess.run('W3:0')
b3 = sess.run('b3:0')

sess.close()
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

my_image_prediction = predict(my_image, parameters)

plt.imshow(image)

plt.show()

print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))