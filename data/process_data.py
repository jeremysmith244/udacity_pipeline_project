import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Takes input files and returns dataframe

    INPUT:
    messages_filepath -- path to the messages csv
    categories_filepath -- path to the categories csv

    OUPUT:
    df -- a pandas datframe containing the merge of the files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''Takes dataframe and cleans formating, returns cleaned df

    INPUT:
    df -- the dataframe containing categories and messages

    OUPUT:
    df -- the dataframe after extracting numerical outputs and labelling
    '''
    # split the categories out into indvidual columns
    categories = df['categories'].str.split(';', expand=True)

    # extract the column names from the first row, and overwrite column labels
    row = categories.iloc[0, :]
    category_colnames = row.str.split('-', expand=True)[0]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop old categories, replace with cleaned ones
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''Takes clean dataframe and saves dataframe to sql database

    INPUT:
    df -- the cleaned pandas dataframe
    database_filename -- path to save the database file

    OUTPUT:
    None
    '''
    # make conneciton to sql, save data as messages table
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('messages', con=engine, index=False,
              index_label='id', if_exists='replace')
    pass


def main():
    '''Solicit user input to run open, clean, save sequence

    INPUT:
    None

    OUTPUT:
    None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
