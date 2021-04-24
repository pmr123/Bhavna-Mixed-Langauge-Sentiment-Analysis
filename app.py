from flask import Flask, render_template, flash, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# App config.
app = Flask(__name__, template_folder='html')

def predict(data):
    tokenizer = Tokenizer(num_words=2500,split=' ')
    tokenizer.fit_on_texts(data)

    X= tokenizer.texts_to_sequences(data)
    X = pad_sequences(X,maxlen=50)

    bimodel = tf.keras.models.load_model('bimodel.h5')
    clmodel = tf.keras.models.load_model('clmodel.h5')
    model = tf.keras.models.load_model('model.h5')
    cnn = tf.keras.models.load_model('cnn.h5')

    x1 = bimodel.predict(X)
    x2 = clmodel.predict(X)
    x3 = model.predict(X)
    x4 = cnn.predict(X)
    ans=[]
    for i in range(len(x1)):
        b=[]
        for j in range(len(x1[i])):
            z = max(x1[i][j],x2[i][j],x3[i][j],x4[i][j] )
            b.append(z)
        ans.append(b)
        b=[]
    
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
            output.append(ans)
    print(output)
    return render_template("index.html", output=output)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print("Running on ",len(gpus)," GPUs")
    app.run(port=5000, debug=True)