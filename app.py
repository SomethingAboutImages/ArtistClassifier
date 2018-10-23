from flask import Flask, send_from_directory, request
import matplotlib.pyplot as plt
import base64
import codecs
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re, os, sys
from load import *

sys.path.append(os.path.abspath('./model'))
app = Flask(__name__, static_folder='frontend/build')
model, graph = init()


def convertImage(imgData1):
    pat = re.compile(b'base64,(.*)')
    imgstr = re.search(pat, imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(codecs.decode(imgstr, 'base64'))


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != '' and os.path.exists('frontend/build/' + path):
        return send_from_directory('frontend/build', path)
    else:
        return send_from_directory('frontend/build', 'index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)

    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    # plt.imshow(x)
    # plt.show()
    # imshow(x)
    x = x.reshape(1, 28, 28, 1)
    with open('file', 'wb') as output:
        output.write(x)

    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
