import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imshow, imresize
import tensorflow as tf
import urllib
import os

RESNET50_URL = 'http://modeldepot.io/assets/uploads/models/models/2fefdb45-9b31-45c6-a714-dc76f8576c6b_resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def init():
    models = {}
    sizes = {}

    # Base ResNet50 model
    models['resnet50'] = initResNet50()
    sizes['resnet50'] = (224, 224)

    # Picasso - Not Picasso
    models['picasso'] = initPicasso()
    sizes['picasso'] = (224, 224)

    graph = tf.get_default_graph()

    return models, sizes, graph

def initResNet50():
    print('Loading ResNet50 model...')
    if not os.path.isfile('resnet50.h5'):
        urllib.urlretrieve(RESNET50_URL, filename='resnet50.h5')
        print('ResNet50 model downloaded')
    model = ResNet50(weights='resnet50.h5')
    print('ResNet50 model loaded from disk')

    return model

def initPicasso():
    print('Loading Picasso model...')
    model = load_model('picasso_overfit.h5')
    print('Picasso model loaded from disk')

    return model