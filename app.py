from flask import Flask, render_template, request
from scipy.misc.pilutil import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import base64
sys.path.append(os.path.abspath("./model"))
from load import *
from keras.preprocessing import image

global graph, model

model, graph = init()

app = Flask(__name__)


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.files.get('file')
    imgData.save("./"+"output.png")
    path = './output.png'
    img=image.load_img(path, grayscale=False, color_mode='rgb',target_size=(28, 28))
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    with graph.as_default():
        out = model.predict(images)
        print(out)
        print(np.argmax(out, axis=1))

        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    app.run(debug=True)
