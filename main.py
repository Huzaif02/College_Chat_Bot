# from spacy.lang.en import English
# import numpy
# from flask import Flask, render_template, request
# import json
# import pickle
# import os
# import time
# import tensorflow as tf
# from tensorflow.keras import layers, models, regularizers
# from voc import voc
# import random

# nlp = English()
# tokenizer = nlp.Defaults.create_tokenizer(nlp)
# PAD_Token=0

# app = Flask(__name__)
     
# model= models.load_model('mymodel.h5')
        
# with open("mydata.pickle", "rb") as f:
#     data = pickle.load(f)


# def predict(ques):
#     ques= data.getQuestionInNum(ques)
#     ques=numpy.array(ques)
#    # ques=ques/255
#     ques = numpy.expand_dims(ques, axis = 0)
#     y_pred = model.predict(ques)
#     res=numpy.argmax(y_pred, axis=1)
#     return res
    

# def getresponse(results):
#     tag= data.index2tags[int(results)]
#     response= data.response[tag]
#     return response

# def chat(inp):
#     while True:
#         inp_x=inp.lower()
#         results = predict(inp_x)
#         response= getresponse(results)
#         return random.choice(response)

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/get")
# def get_bot_response():
#     userText = request.args.get('msg')
#     return str(chat(userText))

# if __name__ == "__main__":
#         app.run()
 

import numpy as np
from flask import Flask, render_template, request
import pickle
import random
import tensorflow as tf
from tensorflow.keras import models
from voc import voc  # Ensure Voc class is imported from voc.py
import nltk
from nltk.tokenize import word_tokenize

# Download punkt tokenizer if not already downloaded
nltk.download('punkt')

# Placeholder token for padding
PAD_Token = 0

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model = models.load_model('mymodel.h5')

# Load preprocessed data
with open("mydata.pickle", "rb") as f:
    data = pickle.load(f)

def predict(ques):
    ques = data.getQuestionInNum(ques)
    print(f"Numerical representation of question: {ques}")  # Debugging line
    ques = np.array(ques)
    ques = np.expand_dims(ques, axis=0)
    y_pred = model.predict(ques)
    print(f"Model prediction: {y_pred}")  # Debugging line
    res = np.argmax(y_pred, axis=1)
    print(f"Predicted tag index: {res}")  # Debugging line
    return res

def getresponse(results):
    tag = data.index2tags[int(results)]
    response = data.response[tag]
    print(f"Selected response tag: {tag}, Response: {response}")  # Debugging line
    return response

def chat(inp):
    inp_x = inp.lower()
    results = predict(inp_x)
    response = getresponse(results)
    return random.choice(response)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(chat(userText))

if __name__ == "__main__":
    app.run()
