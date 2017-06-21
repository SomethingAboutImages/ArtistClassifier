import keras
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf


# def make_parallel(model, gpu_count):
#     def get_slice(data, idx, parts):
#         shape = tf.shape(data)
#         size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
#         stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
#         start = stride * idx
#         return tf.slice(data, start, size)
#
#     outputs_all = []
#     for i in range(len(model.outputs)):
#         outputs_all.append([])
#
#     # Place a copy of the model on each GPU, each getting a slice of the batch
#     for i in range(gpu_count):
#         with tf.device('/gpu:%d' % i):
#             with tf.name_scope('tower_%d' % i) as scope:
#
#                 inputs = []
#                 # Slice each input into a piece for processing on this GPU
#                 for x in model.inputs:
#                     input_shape = tuple(x.get_shape().as_list())[1:]
#                     slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
#                     inputs.append(slice_n)
#
#                 outputs = model(inputs)
#
#                 if not isinstance(outputs, list):
#                     outputs = [outputs]
#
#                 # Save all the outputs for merging back together later
#                 for l in range(len(outputs)):
#                     outputs_all[l].append(outputs[l])
#
#     # merge outputs on CPU
#     with tf.device('/cpu:0'):
#         merged = []
#         for outputs in outputs_all:
#             merged.append(merge(outputs, mode='concat', concat_axis=0))
#
#         return Model(input=model.inputs, output=merged)


batch_size = 128
num_classes = 10
epochs = 50

img_rows, img_columns = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_columns)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_columns)
    input_shape = (1, img_rows, img_columns)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_columns, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_columns, 1)
    input_shape = (img_rows, img_columns, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# tbCallBack.set_model(model)

# model = make_parallel(model, 2)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tbCallBack])

# plot_model(model, 'model.png')

score = model.evaluate(x_test, y_test)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
