import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, make_scorer, fbeta_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])


def tokenize(text):

    # normalize text to lowercase, drop punctuation
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # extract and replace urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

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


def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)

    # split into X an y, apply tokenizer to X data
    X = df['message'].apply(tokenize)
    y = df.iloc[:, 4:]

    # summarize categories
    category_names = y.columns

    return X, y, category_names


def build_model():

    # define parameters for grid search
    parameters = {
        'vect__stop_words': ['english'],
        'vect__max_features': [1000, 5000, 20000],
        'vect__max_df': [0.75, 1.0],
        'clf__n_estimators': [10],
        'clf__min_samples_split': [8, 32],
        'clf__max_depth': [None],
        'clf__criterion': ['gini']
    }

    # create scorer to select recall biased model
    scorer = make_scorer(fbeta_score, beta=10)

    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    # wrap pipeline in grid search
    model = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring=scorer,
        n_jobs=4)

    return model


def evaluate_model(model, x_test, y, category_name):
    # get only the category to be predicted for this model and predict
    y_test = y[category_name]
    y_pred = model.predict(x_test)
    labels = np.unique(y_pred)

    # evaluate the confusion matrix, f10, accuracy, recall and precision
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()
    fscore = fbeta_score(y_test, y_pred, 10)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # print outputs
    print("Category was:", category_name)
    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("F10 Score:", fscore)
    print("Recall Score:", recall)
    print("Precision Score:", precision)


def save_model(model, model_filepath, category):
    # save file for this model using pickle
    with open(model_filepath + '_' + category + '.pkl', 'wb') as picklefile:
        pickle.dump(model, picklefile)


def main():
    # make sure appropriate input passed, load data and split
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        # loop through all the models for each possible prediction, fit, output
        for category in category_names:

            if Y_train[category].unique().shape[0] == 2:
                print('Building model for:', category)
                model = build_model()

                print('Training model for:', category)
                model.fit(X_train, Y_train[category])
                best_params = model.best_params_
                print('Best parameters were \n', best_params)

                print('Evaluating model for:', category)
                evaluate_model(model, X_test, Y_test, category)

                print('Saving model...\n    MODEL: {}'.format(model_filepath))
                save_model(model, model_filepath, category)

                print('Trained model for {} saved!'.format(category))

            else:
                print('Train data for {} does not have two categories'.format(category))

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
