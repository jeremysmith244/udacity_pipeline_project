import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, fbeta_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])


def tokenize(text):
    '''Takes string, normalizes, tokenizes, lemmatizes and returns string

    INPUT:
    text -- a text string to transformed

    OUPUT:
    clean_tokens -- the cleaned, tokenized and lemmatized list of words
    '''
    # normalize text to lowercase, drop punctuation
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize tokens
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def load_data(database_filepath):
    '''Connect to db, extract and clean X, y and categories for training

    INPUT:
    database_filepath -- path to sql database with messages and categories

    OUPUT:
    X -- text inputs for training models
    y -- dataframe with all the categories to be trained
    category_names -- the labels for the categories
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)

    # split into X an y, apply tokenizer to X data
    X = df['message']
    y = df.iloc[:, 4:]

    # summarize categories
    category_names = y.columns

    return X, y, category_names


def build_model():
    '''Build gridsearch model around tfidf transformed randomforest

    INPUT:
    None

    OUPUT:
    pipeline -- multioutput gridsearch random forest pipeline
    '''
    # define parameters for grid search
    parameters = {
        'vect__stop_words': ['english'],
        'vect__max_features': [1000, 5000, 10000],
        'vect__max_df': [0.7, 1.0],
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [8, 11, 16],
        'clf__estimator__max_depth': [None],
        'clf__estimator__criterion': ['gini']
    }

    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # make recall biased scorer
    scorer = make_scorer(fbeta_score, beta=3, average='macro')

    # wrap pipeline in grid search
    model = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=scorer,
        n_jobs=4)

    return model


def test_model(model, y_test, X_test):
    '''Runs classification report on all categories

    INPUT:
    model -- multioutput gridsearch randomforest pipeline
    y_test -- known data for testing
    X_test -- input for model to predict

    OUPUT:
    None
    '''
    y_preds = model.predict(X_test)

    for i, column in enumerate(y_test):

        true_val = y_test[column]
        pred_val = y_preds[:, i]
        print('Score for {}: \n'.format(column))
        print(classification_report(true_val, pred_val))
        print(40*'-')

    pass


def save_model(model, model_filepath):
    '''Save model as pickle file

    INPUT:
    model -- trained model
    model_filepath -- location ot save the model
    '''
    # save file for this model using pickle
    with open(model_filepath + '.pkl', 'wb') as picklefile:
        pickle.dump(model, picklefile)
    pass


def main():
    '''Solicit user input to run build, train, evaluate sequence

    INPUT:
    None

    OUPUT:
    None
    '''
    # make sure appropriate input passed, load data and split
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        for column in Y_train.columns:
            if Y_train[column].unique().shape[0] != 2:
                Y_train.drop(columns=column, inplace=True)
                Y_test.drop(columns=column, inplace=True)
                print('Drop {} because it is not binary'.format(column))

        # loop through all the models for each possible prediction, fit, output
        print(40*'-')
        print('Fitting model with grid search')
        model = build_model()
        model.fit(X_train, Y_train)

        best_params = model.best_params_
        print('Best parameters were: \n')
        print(best_params)

        test_model(model, Y_test, X_test)

        print('Saving model')
        save_model(model, model_filepath)
        with open(model_filepath + '_columns', 'w') as f:
            for column in Y_train.columns:
                f.write(column + '\n')
        print('Model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
