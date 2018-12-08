## Web Image Classifier

This project was created by the Something About Images group for the Fall 2018 CSC 480 Artificial Intelligence class at Cal Poly SLO.

[Try the demo](https://dev.christianjohansen.com/artists/)

[Kaggle notebook that trains the model](https://www.kaggle.com/awallst/somethingaboutimages-csc-480-cal-poly)

__Note: This is a very basic version right now__

### Backend:

```
Need packages like keras, tensorflow, etc. Look in app.py

python2 app.py

This will start the Flask server and download the ResNet50 model if it does not exist.
```

### Frontend:

Install dependencies:
```
cd frontend && yarn
```

#### For development: 
```
yarn start
```
This will open up the Create React App project for auto reloading. Classifying will not work, need to run `yarn build` (CORS stuff I don't care about now)

#### For production: 
```
yarn build
```
Flask server will serve static files from frontend/build/
