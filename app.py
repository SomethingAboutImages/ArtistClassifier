from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import base64
import codecs
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re, os, sys
import better_exceptions
from load import *

sys.path.append(os.path.abspath('./model'))

app = Flask(__name__)

model, graph = init()


def convertImage(imgData1):
    pat = re.compile(b'base64,(.*)')
    imgstr = re.search(pat, imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(codecs.decode(imgstr, encoding='base64'))


@app.route('/')
def index():
    # initModel()
    return render_template("index.html")


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

    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
