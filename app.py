from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import base64
import codecs
from scipy.misc import imsave, imread, imresize
import time
import numpy as np
import keras.models
import re, os, sys
import json
from load import *

sys.path.append(os.path.abspath('./model'))
app = Flask(__name__, static_folder='frontend/build')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model, graph = init()

def sendError(message):
    return jsonify({'status': 'ERROR', 'message': message})

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convertImage(img_data):
    with open('output.png', 'wb') as output:
        output.write(img_data)

@app.route('/predict/', methods=['POST'])
def predict():
    # Check for file called 'file'
    if 'file' not in request.files:
        return sendError('No "file" part')
    # Get file
    file = request.files['file']
    # Check for a filename
    if file.filename == '':
        return sendError('No selected file')

    convertImage(file.read())

    # Check that the extension is allowed
    # If so, save to uploads folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(time.time()) + '-' + filename))
    else:
        return sendError('Invalid filetype')

    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    with open('file', 'wb') as output:
        output.write(x)

    with graph.as_default():
        out = model.predict(x)
        return jsonify({'status': 'SUCCESS', 'response': out.tolist()[0]})

# React app static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != '' and os.path.exists('frontend/build/' + path):
        return send_from_directory('frontend/build', path)
    else:
        return send_from_directory('frontend/build', 'index.html')

if __name__ == '__main__':
    # Create uploads directory
    if not os.path.exists('uploads'):
        os.mkdir('uploads')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
