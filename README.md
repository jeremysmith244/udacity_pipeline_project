# Disaster Response Pipeline Project

<<<<<<< HEAD
## Outline
This program takes data from Figure Eight, which are messages during a disaster
response, with labelled categories, cleans the messages using a an ETL pipeline
and then fits a RandomForestClassifier using a grid search. It then outputs
the results of the classification as a dashboard.

There are three python functions important to this project. One which does the
cleaning and building of a sql database, one which fits the supervised learning
model, and one which controls the flask website. These are described below.

Note that the fitting is heavily biased towards recall (using an F10 score to train the grid search), since in a disaster situation you would rather not miss many important messages.

||||||| 8198187... updated doctrings and readme
## Outline
This program takes data from Figure Eight, which are messages during a disaster
response, with labelled categories, cleans the messages using a an ETL pipeline
and then fits a RandomForestClassifier using a grid search. It then outputs
the results of the classification as a dashboard.

There are three python functions important to this project. One which does the
cleaning and building of a sql database, one which fits the supervised learning
model, and one which controls the flask website. These are described below.

=======
>>>>>>> parent of 8198187... updated doctrings and readme
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
<<<<<<< HEAD
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier, summarizes fit quality and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier`
    - After running this command the models will exist in models folder
||||||| 8198187... updated doctrings and readme
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier, summarizes fit quality and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - After running this command the models will exist in models folder
=======
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
>>>>>>> parent of 8198187... updated doctrings and readme

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
