import pickle
from flask import Flask
from .model import predict

# Initialize flask object
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='\xe0\xcd\xac#\x06\xd9\xe4\x00\xa5\xf2\x88\xc3\xef$\xa5\x05n\x97\xd8\x1269i\xd3'
)

from flask import (
    redirect, render_template, request, session, url_for
)

#Load logistic regression model
with open('./flaskr/static/LogisticRegression.pickle', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method=='POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("base.html")


@app.route('/result', methods=('GET', 'POST'))
def result():
    message = session.get('message')
    df_pred = predict(model=model, text=message)
    sentiment = df_pred.head(1)['sentiment'].values[0]
    score = df_pred.head(1)['score'].values[0]
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("result.html", message=message, sentiment=sentiment, score=score)