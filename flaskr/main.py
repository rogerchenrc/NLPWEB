import pickle
import dill
from flask import Flask
from .model import predict  # Import predict function from model.py
from .tempmodel import predict as tmppredict
from .Model_Loader import text_to_genres
'''
Initiate a new flaskr app
1. Input some random secret key to be used by the application 
2. Input some flaskr commands that would be used by the application
'''
app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='\xe0\xcd\xac#\x06\xd9\xe4\x00\xa5\xf2\x88\xc3\xef$\xa5\x05n\x97\xd8\x1269i\xd3'
)
from flask import (
    redirect, render_template, request, session, url_for
)

'''
Load the machine learning libraries 
1. Logistic regression model is used to predict the sentiment on the newly computed matrix
'''

# Load the machine learning model
# with open('./flaskr/static/model.pkl', 'rb') as input_file:
#     model = pickle.load(input_file)
# with open('./flaskr/static/mlb.pkl', 'rb') as input_file:
#     mlb = pickle.load(input_file)
# with open('./flaskr/static/tfidf.pkl', 'rb') as input_file:
#     tfidf = pickle.load(input_file)

# # Load the text preprocessor transformer
# text_preprocessor = pickle.load(open("text_preprocessor.pickle", 'rb'))
# # Load the multi-hot binary encoder
# binary_encoder = pickle.load(open(binary_encoder.pickle, 'rb'))
# # Load TorchText TEXT field
# TEXT = dill.load(open(TEXT_field_file, "rb"))
# # Load the model parameters
# model_kwargs = pickle.load(open(model_kwargs_file, 'rb'))

## Our final model
with open('./flaskr/static/text_preprocessor.pickle', 'rb') as input_file:
    text_preprocessor = pickle.load(input_file)
with open('./flaskr/static/binary_encoder.pickle', 'rb') as input_file:
    binary_encoder = pickle.load(input_file)
with open('./flaskr/static/model_kwargs.pickle', 'rb') as input_file:
    model = pickle.load(input_file)
with open('./flaskr/static/trained_model.pt', 'rb') as input_file:
    model_weight = pickle.load(input_file)
with open('./flaskr/static/Text.Field', 'rb') as input_file:
    TEXT = dill.load(input_file)
'''
Home Page
1. It will take both GET and POST requests 
2. For GET request, base.html (homepage) will be rendered without any results shown
3. For POST request, input message will be obtained from the form in base.html.
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    b) The page will then be redirected to /result page
'''


@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("base.html")


'''
Result Page
1. It will take both GET and POST requests 
2. For GET request, 'message' will be obtained from the session, remember the 'message' is from the Home page! 
    a) Sentiment and its score(probability) will be predicted by passing in the vectorizer (optional), model and message from the session
    b) The result page will then be rendered based on the message, sentiment and score computed by the predictions
3. For POST request, input message will be obtained from the form in result.html 
    a) Session will then be cleared (to remove anything belonged to previous session) and 'message' will be passed into the session 
    so that it can be reused throughout the session
    c) The page will then be redirected to /result page
'''


@app.route('/result', methods=('GET', 'POST'))
# def result():
#     message = session.get('message')
#     pred = tmppredict(text=message, model=model, mlb=mlb, vectorizer=tfidf)
#     genre = pred.head(1)['genre'].values[0]
#     score = pred.head(1)['score'].values[0]
#     if request.method == 'POST':
#         message = request.form['message']
#         if message is not None:
#             session.clear()
#             session['message'] = message
#             return redirect(url_for('result'))
#     return render_template("result.html", message=message, sentiment=genre, score = score)
def result():
    message = session.get('message')
    df_pred = text_to_genres(text=message, label_threshold=0.5, model_kwargs=model,
                   model_weights=model_weight, binary_encoder=binary_encoder,
                   TEXT=TEXT, text_preprocessor=text_preprocessor)
    genre = df_pred.head(1)['genre'].values[0]
    score = df_pred.head(1)['score'].values[0]
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("result.html", message=message, sentiment=genre, score=score)
