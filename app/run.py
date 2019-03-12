import json
import plotly
import pandas as pd
from glob import glob
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)


def tokenize(text):
    '''Takes string, normalizes, tokenizes, lemmatizes and returns string'''
    # normalize text to lowercase, drop punctuation
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = ''
    for tok in tokens:

        # lemmatize tokens
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens = clean_tokens + clean_tok + ' '

    return clean_tokens


# connect to sql database, get data into pandas dataframe
engine = create_engine('sqlite:///{}'.format('data/DisasterResponse.db'))
df = pd.read_sql_table('messages', con=engine)
X = df['message'].apply(tokenize)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''Generates plotly visualizations for website'''
    # extract data needed for visual 1
    data = df.iloc[:, 4:]
    genre_counts = data.sum().values
    genre_names = data.columns

    # apply countvectorizer and pull top 20 words for visual 2
    count = CountVectorizer(stop_words='english')
    words = count.fit_transform(X)
    sum_words = words.sum(axis=0)
    words_freq_dict = [(word, sum_words[0, idx]) for word,
                       idx in count.vocabulary_.items()]
    words_freq_dict = sorted(words_freq_dict, key=lambda x: x[1], reverse=True)
    top_words_dict = words_freq_dict[0:20]

    top_words = []
    words_freq = []
    for tup in top_words_dict:
        top_words.append(tup[0])
        words_freq.append(tup[1])

    graphs = [
        # create plotly graph showing number of messages in each category
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # create plotly graph showing top 20 words in corpus
        {
            'data': [
                Bar(
                    x=top_words,
                    y=words_freq
                )
            ],

            'layout': {
                'title': 'Top 20 Words in Corpus',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # load model
    model_path = glob('models/*.pkl')[0]
    classification_results = {}

    model = joblib.load(model_path)
    predictions = model.predict([query])[0]

    # load column labels
    with open('models/classifier_columns') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            classification_results[line] = predictions[i]

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
