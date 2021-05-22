from flask import Flask, render_template, request
import jsonify
import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
# load model and transformer
model = load_model('best_model.h5')
sc = joblib.load('scaler.pkl')

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        fa = float(request.form['fa'])
        va = float(request.form['va'])
        ca = float(request.form['ca'])
        rs = float(request.form['rs'])
        chl = float(request.form['chl'])
        fsd = float(request.form['fsd'])
        tsd = float(request.form['tsd'])
        d = float(request.form['d'])
        ph = float(request.form['ph'])
        s = float(request.form['s'])
        a = float(request.form['a'])
        prediction = np.argmax(model.predict(sc.transform([[fa,va,ca,rs,chl,fsd,tsd,d,ph,s,a]])),axis=-1)
        output=prediction.tolist()

        if output==[0]:
            return render_template('index.html',prediction_text="wine quality rating is 3")
        elif output==[1]:
            return render_template('index.html',prediction_text="wine quality rating is 4")
        elif output==[2]:
            return render_template('index.html',prediction_text="wine quality rating is 5")
        elif output==[3]:
            return render_template('index.html',prediction_text="wine quality rating is 6")
        elif output==[4]:
            return render_template('index.html',prediction_text="wine quality rating is 7")
        else:
            return render_template('index.html',prediction_text="wine quality rating is 8")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

