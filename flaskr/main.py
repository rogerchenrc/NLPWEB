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

# Find the project directory
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
- Takes both GET and POST requests 
- For GET request, a film description will be obtained from the session 
    a) The genre and its score (probability) will be predicted
    b) The result page will then be rendered to display the description, predicted genre(s) and probabilities
- For POST request, the input description will be obtained from the form in result.html 
    a) Session will then be cleared (to remove anything belonged to previous session) and the description will be passed 
    into the session so that it can be reused throughout the session
    b) The page will then be redirected to /result page
'''


@app.route('/result', methods=('GET', 'POST'))

def result():
    # Get the text from the search bar
    message = session.get('message')
    
    # Made a prediction from the model
    df_pred = text_to_genres(message,
                             model_kwargs_file=root_path+'\\static\\model_kwargs.pickle',
                             model_weights_file=root_path+'\\static\\trained_model.pt',
                             binary_encoder_file=root_path+'\\static\\binary_encoder.pickle',
                             TEXT_field_file=root_path+"\\static\\TEXT.Field",
                             text_preprocessor_file=root_path+"\\static\\text_preprocessor.pickle")
    
    genre = df_pred.head(1)['genre'].values[0]
    score = df_pred.head(1)['score'].values[0]
    
    if request.method == 'POST':
        message = request.form['message']
        if message is not None:
            session.clear()
            session['message'] = message
            return redirect(url_for('result'))
    return render_template("result.html", message=message, genre=genre, score=score)

if __name__ == "__main__":
    app.run()
