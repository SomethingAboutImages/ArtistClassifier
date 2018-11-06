from keras.applications.resnet50 import decode_predictions

def decodeArtists(preds):
    preds = preds[0]
    print(preds)
    labels = ['Ivan Aivazovsky', 'Gustave Dore', 'Odilon Redon', 'Rembrandt', 'Edgar Degas', 'Claude Monet',
              'Albrecht Durer', 'Francisco Goya', 'Theophile Steinlen', 'Ivan Shishkin', 'Giovanni Battista Piranesi',
              'Camille Corot', 'Pierre-Auguste Renoir', 'Childe Hassam', 'Raphael Kirchner', 'James Tissot',
              'Alfred Sisley', 'Paul Cezanne', 'John Singer Sargent', 'Vincent van Gogh', 'Zdislav Beksinski',
              'Camille Pissarro', 'Eugene Boudin', 'Fernand Leger', 'Boris Kustodiev', 'Nicholas Roerich',
              'Ilya Repin', 'Martiros Saryan', 'Isaac Levitan', 'Pyotr Konchalovsky', 'Salvador Dali',
              'Pablo Picasso', 'Henri Matisse', 'Marc Chagall', 'Erte', 'Paul Gauguin', 'Eyvind Earle',
              'Zinaida Serebriakova']
    i = 0
    output = []
    for label in labels:
        output.append({'label': label, 'value': float(preds[i])})
        i += 1
    output = sorted(output, key=lambda x: x['value'], reverse=True)
    return output[:5]

def decodeResNet50(preds):
    output = [{'label': str(t[1]), 'value': float(t[2])} for t in decode_predictions(preds, top=5)[0]]
    return output

def decodePicasso(preds):
    output = [{'label': 'Picasso', 'value': float(preds[0][1])}, {'label': 'Not Picasso', 'value': float(preds[0][0])}]
    return output

def decodePicassoOneEpoch(preds):
    output = [{'label': 'Picasso', 'value': float(preds[0][1])}, {'label': 'Not Picasso', 'value': float(preds[0][0])}]
    return output