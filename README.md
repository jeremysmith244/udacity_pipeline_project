# Disaster Response Pipeline Project

## Outline
This program takes data from Figure Eight, which are messages during a disaster
response, with labelled categories, cleans the messages using a an ETL pipeline
and then fits a RandomForestClassifier using a grid search. It then outputs
the results of the classification as a dashboard.

There are three python functions important to this project. One which does the
cleaning and building of a sql database, one which fits the supervised learning
model, and one which controls the flask website. These are described below.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier, summarizes fit quality and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - After running this command the models will exist in models folder

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
