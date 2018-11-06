from flask import Flask, send_from_directory, request, jsonify
from werkzeug.utils import secure_filename
import base64
from scipy.misc import imsave, imread, imresize
import time
import numpy as np
import keras.models
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import json
from load import *

app = Flask(__name__, static_folder='frontend/build')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Initialize all the models
models, sizes, decode, graph = init()

def sendError(message):
    res = jsonify({'status': 'ERROR', 'message': message})
    res.status_code = 400
    return res

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def tempSaveImage(name, img_data):
    filename = app.config['UPLOAD_FOLDER'] + '/' + str(time.time()) + '_' + secure_filename(name)
    with open(filename, 'wb') as output:
        output.write(img_data)
    return filename

@app.route('/predict/<modelName>', methods=['POST'])
def predict(modelName):
    # Check that model exists
    if modelName not in models:
        return sendError('Invalid model')
    # Check for file called 'file'
    if 'file' not in request.files:
        return sendError('No "file" part')
    # Get file
    file = request.files['file']
    # Check for a filename
    if file.filename == '' or not allowed_file(file.filename):
        return sendError('Invalid file name')

    imgfile = tempSaveImage(file.filename, file.read())

    img = image.load_img(imgfile, target_size=sizes[modelName])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    with graph.as_default():
        preds = models[modelName].predict(x)
        output = decode[modelName](preds)
        print(output)
        return jsonify({'status': 'SUCCESS', 'response': output})

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
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.mkdir(app.config['UPLOAD_FOLDER'])

    port = int(os.environ.get('PORT', 5100))
    app.run(host='127.0.0.1', port=port)
