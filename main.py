'''
This module is to look at what a pipeline woud look like
for different trained models and compare them for different datasets
'''
import argparse
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, evaluate_model

class ReadCsv:
    '''Read the csv file class.'''
    def __init__(self, path, lst_features, target, model_type):
        '''Initialize the class with the path, list of features and the target column.
        
        :param path: str: path to the csv file
        :param lst_features: list: list of features
        :param target: str: target column
        '''

        self.lst_features = lst_features
        self.path = path
        self.target = target
        self.model_type = model_type
    def train_model(self) -> pd.DataFrame:
        '''Read the csv file and return the features and the target column.

        Accepts no arguments.
        Returns: pd.DataFrame: features and the target column.
        rtype: pd.DataFrame, str
        '''
        return pd.read_csv(self.path)[self.lst_features], self.target, self.model_type

class Model:
    '''Model class to train the model.'''
    def __init__(self, csv: ReadCsv):
        '''Initialize with the csv file.
        accept: read_csv: class for making csv file
        type: read_csv
        '''
        self.csv = csv
    def train_model(self) -> pd.DataFrame:
        '''train model function to returns the features and the target column.

        Accepts no arguments.
        Returns: pd.DataFrame: features and the target column.
        rtype: pd.DataFrame, str
        '''
        return self.csv.train_model()

class Train(Model):
    '''Train class to train the model.'''
    def train_model(self):
        '''Train model function to train the model.
        
        Accepts no arguments.
        Returns: the best model given by pycaret.
        rtype: pycaret.classification or pycaret.regression
        '''
        df_csv, target, model_type = super().train_model()
        df_csv.dropna(inplace=True)
        setup(df_csv, target = target,
              experiment_name= 'admit',
              session_id = 123)
        final_model = create_model(model_type)
        fully_tuned_model = tune_model(final_model)
        evaluate_model(fully_tuned_model)
        return None

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('path', type=str,
                     help = '.data/admit.csv; .data/ebay.csv')
    arg.add_argument('features', type=list,
                     help = 'GPA,SAT Score,Admission Decision; seller_price,ship_price')
    arg.add_argument('target', type=str,
                     help = 'Admission Decision; ship_price')
    arg.add_argument('model', type=str,
                     help = 'ada; huber')
    args = arg.parse_args()
    df = ReadCsv(args.path, args.features, args.target, args.model)
    model = Train(df)
    model.train_model()
    