import numpy as np
from tensorflow.python.keras.models import load_model, Sequential, model_from_json, Model
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imshow, imresize
import tensorflow as tf
import urllib
import os
import random

from decode import decodeArtists, decodeResNet50, decodePicasso, decodePicassoOneEpoch

RESNET50_URL = 'http://modeldepot.io/assets/uploads/models/models/2fefdb45-9b31-45c6-a714-dc76f8576c6b_resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def init():
    models = {}
    sizes = {}
    decode = {}

    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    models['artists'] = initArtists()
    sizes['artists'] = (224, 224)
    decode['artists'] = decodeArtists

    # Base ResNet50 model
    models['resnet50'] = initResNet50()
    sizes['resnet50'] = (224, 224)
    decode['resnet50'] = decodeResNet50

    # Picasso - Not Picasso
    models['picasso'] = initPicasso()
    sizes['picasso'] = (224, 224)
    decode['picasso'] = decodePicasso

    models['picasso_one'] = initPicassoOneEpoch()
    sizes['picasso_one'] = (224, 224)
    decode['picasso_one'] = decodePicassoOneEpoch

    graph = tf.get_default_graph()

    return models, sizes, decode, graph

def initArtists():
    print('Loading Artists model...')
    full_model = load_model('painters_adam.h5')
    print('Artists model loaded from disk')
    
    return full_model

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
    model = load_model('picasso_models/picasso_overfit.h5')
    print('Picasso model loaded from disk')

    return model

def initPicassoOneEpoch():
    print('Loading Picasso one epoch model...')
    model = load_model('picasso_models/picasso_one_epoch.h5')
    print('Picasso one epoch loaded')
    return model
