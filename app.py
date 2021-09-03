from flask import Flask, render_template, flash, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json


# App config.
app = Flask(__name__, template_folder='html')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

bimodel = tf.keras.models.load_model('bimodel.h5')
clmodel = tf.keras.models.load_model('clmodel.h5')
model = tf.keras.models.load_model('model.h5')
cnn = tf.keras.models.load_model('cnn.h5')

with open('tokenizer_json_string.txt') as json_file:
    tokenizer_json_string = json.load(json_file)

tokenizer = tokenizer_from_json(tokenizer_json_string)

def predict(data):
    X= tokenizer.texts_to_sequences(data)
    X = pad_sequences(X,maxlen=50)

    x1 = bimodel.predict(X)
    x2 = clmodel.predict(X)
    x3 = model.predict(X)
    x4 = cnn.predict(X)
    ans = (x1+x2+x3+x4)/4
    print(ans)
    return ans






@app.route("/", methods=['GET', 'POST'])
def page():
    output = []
    if request.method == "POST":
        data = request.form.get("data")
        data = data.strip()
        data = data.split('\n')
        for i in range(len(data)):
            if data[i][-1] == '\r':
                data[i] = data[i][0:-1]
        response = predict(data)
        for i in range(len(response)):
            ans = {
                "data" : data[i],
                "pred" : response[i],
                "pred_out" : np.argmax(response[i])
            }
            print("sentence => ",data[i],", predicted sentiment => ", np.argmax(response[i]),", predictions => ", response[i])
            output.append(ans)
    return render_template("index.html", output=output)

if __name__ == "__main__":
    print("Running on ",len(gpus)," GPUs")
    app.run(port=5000, debug=True)