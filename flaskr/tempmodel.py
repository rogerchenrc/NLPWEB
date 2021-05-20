import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

with open('/Users/khc/Desktop/NLPWEB/flaskr/static/tfidf.pkl', 'rb') as tf:
    tfidf = pickle.load(tf)

with open('/Users/khc/Desktop/NLPWEB/flaskr/static/model.pkl', 'rb') as mod:
    model = pickle.load(mod)

with open('/Users/khc/Desktop/NLPWEB/flaskr/static/mlb.pkl', 'rb') as multi:
    mlb = pickle.load(multi)


def lower_case(text):
    # For different type of datatype
    if type(text) == list:
        low_text = text.map(lambda x: x.lower())
    elif type(text) == str:
        low_text = text.lower()
    else:
        low_text = text.str.lower()
    return low_text


def remove_punctuation(text):
    removed_punc = re.sub(r'[^\w\s]', '', text)
    return removed_punc


def tokenizer(text):
    return word_tokenize(text)


def untokenizer(token_list):
    return " ".join(token_list)


stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
              'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
              'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
              'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
              'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
              'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
              "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
              "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
              'wouldn', "wouldn't"]
stop_words.extend([',', '.', '?', '/', '!', ])


def remove_stopwords(tokens_list):
    cleaned_tokens = []
    for token in tokens_list:
        if token in stop_words: continue
        cleaned_tokens.append(token)
    return cleaned_tokens


def stemmer(tokens_list):
    stemmed_tokens_list = []
    for i in tokens_list:
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
    return stemmed_tokens_list


def lemmatizer(list_of_tokens):
    lemmatized_token_list = []
    for i in list_of_tokens:
        token = WordNetLemmatizer().lemmatize(i)
        lemmatized_token_list.append(token)
    return lemmatized_token_list


def preprocess(text):
    cleaned_texts = []
    num_texts = len(text)
    lower_string = lower_case(text)
    removed_punc = remove_punctuation(lower_string)
    tokenized_list = tokenizer(removed_punc)
    removed_stopwords = remove_stopwords(tokenized_list)
    # stemmed_words = stemmer(removed_stopwords)
    lemmatized_words = lemmatizer(removed_stopwords)
    back_string = untokenizer(lemmatized_words)
    cleaned_texts.append(back_string)
    return cleaned_texts


def predict(text, model, mlb, vectorizer):
    post_text = preprocess(text)
    vectorized = vectorizer.transform(post_text)
    predict = model.predict(vectorized)
    score = round(np.amax(model.predict_proba(vectorized)), 3)
    predicted_genre = mlb.inverse_transform(predict)
    data = [(text, predicted_genre[0][:], score)]
    df = pd.DataFrame(data, columns=['text', 'genre', 'score'])
    print(df)
    return df

# descrip = "Duke reads the story of Allie and Noah, two lovers who were separated by fate, to Ms Hamilton, an old woman who suffers from dementia, on a daily basis out of his notebook."
#
# predict(descrip, model, mlb, tfidf)