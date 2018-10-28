import numpy as np
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imshow, imresize
import tensorflow as tf
import urllib
import os

RESNET50_URL = 'http://modeldepot.io/assets/uploads/models/models/2fefdb45-9b31-45c6-a714-dc76f8576c6b_resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def init():
    print('Loading MNIST model...')
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    print('MNIST model loaded from disk')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph

def initResNet50():
    print('Loading ResNet50 model...')
    if not os.path.isfile('resnet50.h5'):
        urllib.urlretrieve(RESNET50_URL, filename='resnet50.h5')
        print('ResNet50 model downloaded')
    model = ResNet50(weights='resnet50.h5')
    graph = tf.get_default_graph()
    print('ResNet50 model loaded from disk')

    return model