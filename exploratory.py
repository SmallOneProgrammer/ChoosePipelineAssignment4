"""This module contains functions to explore the datasets 
data/admit.csv, data/promote.csv, data/ebay.csv"""

#import libraries for visualization
import pandas as pd
import altair as alt
from IPython.display import display
from pycaret.regression import setup as setup2, create_model as create_model2
from pycaret.regression import tune_model as tune_model2, evaluate_model as evaluate_model2
from pycaret.regression import compare_models as compare_models2
from pycaret.classification import setup, create_model, tune_model, evaluate_model, compare_models

class Explore:
    """To be used to explore the datasets."""
    def __init__(self, file):
        """Initialize the class with the file path.

        file: str, path to the file
        """
        self.file = file
    def read_csv(self, index = False):
        """Read csv file and return dataframe.

        file: str, name of the file
        type: str, type of the file
        return: dataframe
        """
        if index:
            df = pd.read_csv(self.file, index_col = 0)
        else:
            df = pd.read_csv(self.file)
        features = df.columns
        return df, features

    def drop_na(self, df):
        """Drop rows with missing values.

        Accepts df: dataframe
        Returns df: dataframe
        """
        df = df.dropna()
        return df

    def display_chart(self, df, features, target):
        """Display bar chart for each feature.
        Accepts df: dataframe
                features: list of features
                target: target variable

        Returns None
        """
        alt.data_transformers.disable_max_rows()
        for i in features:
            altchart = alt.Chart(df).mark_bar().encode(x = f'mean({i}):Q', y = target)
            display(altchart)
        return None

    def drop_columns(self, df, columns):
        """Drop columns from the dataframe.

        Accepts df: dataframe
                columns: list of columns to drop
        Returns df: dataframe
        """
        df = df.drop(columns, axis = 1)
        return df

class ExploreTrain:
    """To be used to explore the training and distribution methods."""
    @staticmethod
    def pycaret_explore(df, target, type_model = 'classification'):
        """Use pycaret to explore the data and build a model.

        Accepts df: dataframe
                target: target variable

        Returns None
        """
        #initialize setup
        if type_model == 'classification':
            setup(data = df,
            target = target,
            session_id = 123,
            log_experiment = True,
            experiment_name = target)
            #compare models
            best = compare_models()
            #tune model
            tuned_best = tune_model(best)
            #evaluate model
            return evaluate_model(tuned_best)
        if type_model == 'regression':
            setup2(data = df,
            target = target,
            session_id = 123,
            log_experiment = True,
            experiment_name = target)
            #compare models
            best = compare_models2()
            #tune model
            tuned_best = tune_model2(best)
            #evaluate model
            return evaluate_model2(tuned_best)
    @staticmethod
    def use_specific_model_pycaret(df, target, model, type_model = 'classification'):
        """Use a specific model to build a model.

        Accepts df: dataframe
                target: target variable
                model: str, model to use
        Returns None
        """
        if type_model == 'classification':
            setup(data = df,
                        target = target,
                        session_id = 123,
                        log_experiment = True,
                        experiment_name = target)
            model = create_model(model)
            tuned_model = tune_model(model)
            return evaluate_model(tuned_model)
        else:
            setup2(data = df,
                        target = target,
                        session_id = 123,
                        log_experiment = True,
                        experiment_name = target)
            model = create_model2(model)
            tuned_model = tune_model2(model)
            return evaluate_model2(tuned_model)
    @staticmethod
    def check_distribution(df, columns):
        """Check the distribution of the target variable.

        Accepts df: dataframe
                target: target variable
        Returns None
        """
        alt.data_transformers.disable_max_rows()
        for column in columns:
            altchart = alt.Chart(df).mark_bar().encode(x = column, y = 'count()')
            display(altchart)
            