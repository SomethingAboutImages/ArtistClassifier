import numpy as np
from keras.models import load_model, Sequential, model_from_json, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imshow, imresize
import tensorflow as tf
import urllib
import os
import random

RESNET50_URL = 'http://modeldepot.io/assets/uploads/models/models/2fefdb45-9b31-45c6-a714-dc76f8576c6b_resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def init():
    models = {}
    sizes = {}

    random.seed(42)
    np.random.seed(42)
    tf.set_random_seed(42)

    # Base ResNet50 model
    models['resnet50'] = initResNet50()
    sizes['resnet50'] = (224, 224)

    # Picasso - Not Picasso
    models['picasso'] = initPicassoOther()
    sizes['picasso'] = (224, 224)

    models['picasso_one'] = initPicassoOneEpoch()
    sizes['picasso_one'] = (224, 224)

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
    #model = load_model('picasso_models/picasso_one_epoch.h5')
    json_file = open('picasso_overfit.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('picasso_models/picasso_overfit_weights.h5')
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Picasso model loaded from disk')

    return loaded_model

def initPicassoOther():
    print('Loading Picasso model...')
    top_model = Sequential()
    top_model.add(Dense(2, activation='softmax', input_shape=(2048,)))
    top_model.load_weights('picasso_models/picasso_overfit_weights.h5', by_name=True)

    model = ResNet50(include_top=True, weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.layers.pop()

    last = model.layers[-1].output

    x = Dense(2, activation='softmax', weights=top_model.layers[-1].get_weights())(last)

    full_model = Model(model.input, x)
    print('Picasso model loaded from disk')

    return full_model
    return my_new_model

def initPicassoOneEpoch():
    print('Loading Picasso one epoch model...')
    num_classes = 2
    resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    my_new_model = Sequential()
    my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
    my_new_model.add(Dense(num_classes, activation='softmax'))

    # Say not to train first layer (ResNet) model. It is already trained
    my_new_model.layers[0].trainable = False

    my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    my_new_model.load_weights('picasso_models/picasso_one_epoch_weights.h5', by_name=True)

    return my_new_model
