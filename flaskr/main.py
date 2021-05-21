import pickle
import dill
import os
from flask import Flask
import Model_Loader
from Model_Loader import *

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
'''

# Our final model
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

root_path = os.path.dirname(os.path.abspath(__file__))

# predicted_genres, predicted_scores = text_to_genres("The Avengers and their allies must be willing to sacrifice all in an attempt to defeat the powerful Thanos before his blitz of devastation and ruin puts an end to the universe.",
#                                                     model_kwargs_file=root_path+'\\model\\12_Genres\\model_kwargs.pickle',
#                                                     model_weights_file=root_path+'\\model\\12_Genres\\trained_model.pt',
#                                                     binary_encoder_file=root_path+'\\model\\12_Genres\\binary_encoder.pickle',
#                                                     TEXT_field_file=root_path+"\\model\\12_Genres\\TEXT.Field",
#                                                     text_preprocessor_file=root_path+"\\model\\12_Genres\\text_preprocessor.pickle")
# print(predicted_genres, predicted_scores)
    
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
    
#     df_pred = text_to_genres(message,
#                              model_kwargs_file=root_path+'\\model\\12_Genres\\model_kwargs.pickle',
#                              model_weights_file=root_path+'\\model\\12_Genres\\trained_model.pt',
#                              binary_encoder_file=root_path+'\\model\\12_Genres\\binary_encoder.pickle',
#                              TEXT_field_file=root_path+"\\model\\12_Genres\\TEXT.Field",
#                              text_preprocessor_file=root_path+"\\model\\12_Genres\\text_preprocessor.pickle")
    
    
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

if __name__ == "__main__":
    app.run()
