# Disaster Response Pipeline Project

## Outline

![homepage of app](/page1.jpg)
This program takes data from Figure Eight, which are messages during a disaster
response, with labelled categories, cleans the messages using a an ETL pipeline
and then fits a RandomForestClassifier using a grid search. It then outputs
the results of the classification as a dashboard.

There are three python functions important to this project. One which does the
cleaning and building of a sql database, one which fits the supervised learning
model, and one which controls the flask website. These are described below.

Note that the fitting is heavily biased towards recall, since in a disaster situation you would rather not miss many important messages.

The flask website will allow you to input a query and test the model for yourself!

![classification of app](/page2.jpg)

## Dependencies
json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle

## Files

### File One
data/process_data.py is the ETL pipeline. This takes three arguments.

1. a .csv file which contains a set of messages.
2. the labels for training and testing the classifier.
3. path to output a sql database

This function will read both files, clean and organize the data in one table, and save

### File Two
models/train_model.py is the machine learning pipeline. This takes two arguments.

1. path to database output by file 1.
2. path to output the trained classifier as pickle file

The machine learning pipeline will apply normalization, tokenization and lemmatization, then use a TfidfTransformer, which is fed into a RandomForestClassifier, wrapped in a MultiOutputClassifier.

This trained against 80% of the data, against an f_beta score of 3 to bias towards recall.

Metrics on test data is printed to the terminal.

### File Three

File three is a flask app, which outputs a couple summary graphs, and offers user ability to input text to classfy. If input, it then shows results of that classification.

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier, summarizes fit quality and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier`
    - After running this command the models will exist in models folder

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
