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
        
        class_list = {'a': 0, 'apostrophe': 1, 'b': 2, 'c': 3, 'capitalize': 4, 'colon': 5, 'comma': 6, 'd': 7, 'e': 8, 'exclamation': 9, 'f': 10, 'g': 11, 'h': 12, 'hyphen': 13, 'i': 14, 'j': 15, 'k': 16, 'l': 17, 'm': 18, 'n': 19, 'number': 20, 'o': 21, 'p': 22, 'period': 23, 'q': 24, 'question': 25, 'r': 26, 's': 27, 'semicolon': 28, 'space': 29, 't': 30, 'u': 31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36}
        response = np.array_str(np.argmax(out, axis=1))
        for key,value in class_list.items():
            if response == value:
                result = key
        return result


if __name__ == '__main__':
    app.run(debug=True)
