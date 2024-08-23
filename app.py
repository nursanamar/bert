from flask import Flask, render_template, request

import model


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/hasil", methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        text = request.form.get("text")
        
        prediction = model.predict(text)
        
        return render_template("result.html", result=prediction)
    else:
        return render_template("index.html")