import numpy as np
from PIL import Image
from datetime import datetime
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import pickle
import scipy
from scipy import integrate
from keras.preprocessing import image
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import tensorflow as tf

app = Flask(__name__)

global base_model
base_model = VGG16(weights='imagenet')

global model
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)


def extract_features(img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
    img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
    img = img.convert('RGB')  # Make sure img is color
    x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
    x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
    x = preprocess_input(x)  # Subtracting avg values for each pixel
    feature = model.predict(x)[0]  # (1, 4096) -> (4096, )
    return feature / np.linalg.norm(feature)  # Normalize

class Search_Engine(object):
    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path, "rb") as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    # Tinh khoang cach cosine giua vector truy van va cac vector trong du lieu
    def cos_cdist(self, vector):
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    # Tim kiem du lieu 
    def match(self, img, topn=5):
        features = extract_features(img)
        img_distances = self.cos_cdist(features)
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

searcher = Search_Engine('static/features.pck')
topn = 10


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file_image = request.files['query_img']
        # Save query image
        img = Image.open(file_image)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file_image.filename
        img.save(uploaded_img_path)
        names, match = searcher.match(img, topn=10)
        scores = []
        for i in range(topn):
            scores.append((1-match[i],names[i]))
        return render_template('index.html', query_path=uploaded_img_path,scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    run_with_ngrok(app)
    app.run()

