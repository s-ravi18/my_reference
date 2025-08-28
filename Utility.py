!pip install -qq polars copydf shap xgboost --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-cache-dir

import pandas as pd
import awswrangler as wr
import os
import seaborn as sns
import time,datetime
import numpy as np
import pandas as pd
import re
from datetime import date,timedelta
from datetime import datetime
from functools import reduce
import json
from pandas.tseries.offsets import MonthEnd, MonthBegin
from dateutil.relativedelta import relativedelta
import polars as pl
from copydf import copyDF
from tqdm import tqdm

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns",1000)


import psutil
available = psutil.virtual_memory().available*100/psutil.virtual_memory().total
print("available memory:", available)


import warnings

# Suppress all warnings globally
warnings.filterwarnings('ignore')

# Set NumPy print options to suppress scientific notation
np.set_printoptions(suppress=True, precision=6)

pd.options.display.float_format = '{:.6f}'.format  # Set 6 decimal places globally

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import shap
import polars
import pkg_resources

from xgboost import XGBClassifier
import gc



#Function to reduce df memory usage
def reduce_mem_usage(df, int_cast=True, obj_to_category=True, subset=None):
    start_mem = df.memory_usage().sum() / 1024 ** 2;
    gc.collect()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    cols = subset if subset is not None else df.columns.tolist()
    for col in cols:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            # test if column can be converted to an integer
            treat_as_int = str(col_type)[:3] == 'int'
            if int_cast and not treat_as_int:
                treat_as_int = df[col].dtype == 'int'
            if treat_as_int:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif 'datetime' not in col_type.name and obj_to_category:
            df[col] = df[col].astype('category')
    gc.collect()
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.3f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df



def funct_retry(query_string):
    n_attempts = 0
    flag = 0
    while n_attempts<=5 and flag == 0:
        try:
            df = query(query_string)  
            flag = 1
        except Exception as e:
            print(e)
            n_attempts+=1
            
    return df



## Helper Class for modeling;
class evaluation:
    
    def eda(self, df):
        """
        Performs Exploratory Data Analysis (EDA) on the provided dataframe.
        
        Parameters:
        df : pd.DataFrame
            The input dataframe for analysis.

        Returns:
        None
        """
        unique_realid_count = df['realid'].nunique()
        print(f"Number of unique 'realid': {unique_realid_count}")
        duplicates = df[df.duplicated(keep=False)]
        if not duplicates.empty:
            print("Duplicate records found:")
            print(duplicates)
        else:
            print("No duplicate records found.")
            
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        print("Categorical Columns:")
        for col in categorical_columns:
            print(col)
        print("\nNumerical Columns:")
        for col in numerical_columns:
            print(col)

    ## Series as input
    def decile_probs(self, series, n_bins):
        """
        Calculates decile cutoffs for a given series based on specified number of bins.

        Parameters:
        series : pd.Series
            Input series of probabilities for which deciles are to be determined.
        n_bins : int
            The number of bins (deciles) to create.

        Returns:
        bin_edges : ndarray
            Edges of the bins representing deciles.
        """
        ## Defining the Decile Cutoffs for CIBIL Score;  ## final['cibil_current']
        _, bin_edges = pd.qcut(series, q=n_bins, labels=False, duplicates='drop', retbins=True)
        return bin_edges
    
    ### Takes in Actual and reference probs
    def decile_chart(self, prob, actual, reference_prob):
        """
        Generates a decile summary chart comparing actual values against predicted probabilities.

        Parameters:
        prob : pd.Series
            Series of predicted probabilities.
        actual : pd.Series
            Series of actual outcomes (0s and 1s).
        reference_prob : ndarray
            Array of reference probabilities to define deciles.

        Returns:
        decile_summary : pd.DataFrame
            DataFrame summarizing the decile results.
        """
        # Step 2: Apply oot Deciles to oot/OOT
        df = pd.DataFrame({'Probability': prob, 'Actual': actual})
        # Assign oot records into the same deciles based on oot bin edges
        df['Decile'] = pd.cut(
            df['Probability'],
            bins=reference_prob,
            labels=False,
            include_lowest=True
        )
        
        # Step 3: Aggregate Results for Decile Summary
        decile_summary = df.groupby('Decile').agg(
            Total=('Actual', 'count'),
            Good=('Actual', lambda x: (x == 0).sum()),
            Bad=('Actual', lambda x: (x == 1).sum())
        ).reset_index()
        
        # Add Probability Range Column
        prob_ranges = [f"[{reference_prob[i]:.6f} - {reference_prob[i+1]:.6f}]" for i in range(len(reference_prob)-1)]
        decile_summary['Probability Range'] = prob_ranges[:len(decile_summary)]  # Assign ranges to deciles    
        
        # Compute additional metrics
        decile_summary['Bad Rate'] = decile_summary['Bad'] / decile_summary['Total']
        decile_summary['%Good'] = decile_summary['Good'] / decile_summary['Good'].sum()
        decile_summary['%Bad'] = decile_summary['Bad'] / decile_summary['Bad'].sum()
        decile_summary['%Pop'] = decile_summary['Total'] / decile_summary['Total'].sum()
        
        # Convert to percentage format
        for col in ['Bad Rate', '%Good', '%Bad', '%Pop']:
            decile_summary[col] = decile_summary[col].map(lambda x: f"{x:.1%}")
        
        # Reverse row order for final display
        decile_summary = decile_summary.iloc[::-1].reset_index(drop=True)
        
        # Cumulative Metrics
        decile_summary['Cumm Pop'] = decile_summary['Total'].cumsum() / decile_summary['Total'].sum() * 100
        decile_summary['Cumm Good'] = decile_summary['Good'].cumsum() / decile_summary['Good'].sum() * 100
        decile_summary['Cumm Bad'] = decile_summary['Bad'].cumsum() / decile_summary['Bad'].sum() * 100
        
        # Fix for 'Area' column - Ensure numeric operations happen first
        # Compute Area using trapezoidal approximation
        decile_summary['Area'] = (
            decile_summary['Cumm Good'] - decile_summary['Cumm Good'].shift(1).fillna(0)
        ) * (
            decile_summary['Cumm Bad']
        ) / 100    
        return decile_summary   
    
    
    def calculate_csi(self, df_reference, df_current, bins=10):
        """
        Iteratively calculates the CSI for each numeric column in the dataset,
        handling NaNs, infinite values, and constant columns.

        Parameters:
        df_reference (pd.DataFrame): Reference (historical) dataset.
        df_current (pd.DataFrame): Current (new) dataset.
        bins (int): Number of bins for grouping feature distributions.

        Returns:
        pd.DataFrame: DataFrame with feature names and their CSI scores.
        """
        csi_results = []
        for col in df_reference.columns:
            if df_reference[col].dtype in [np.float64, np.int64]:  # Process only numeric columns
                # Remove NaNs and Infinite values
                ref_col = df_reference[col].replace([np.inf, -np.inf], np.nan).dropna()
                cur_col = df_current[col].replace([np.inf, -np.inf], np.nan).dropna()
                # Skip columns with constant values
                if ref_col.nunique() < 2 or cur_col.nunique() < 2:
                    continue
                # Define bin edges based on reference data
                ref_bins = np.histogram(ref_col, bins=bins)[1]
                # Compute distributions
                ref_dist, _ = np.histogram(ref_col, bins=ref_bins, density=True)
                cur_dist, _ = np.histogram(cur_col, bins=ref_bins, density=True)
                # Normalize to probabilities and avoid division by zero
                ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
                cur_dist = np.where(cur_dist == 0, 0.0001, cur_dist)
                # Compute CSI
                csi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
                csi_results.append({"Feature": col, "CSI": round(csi, 4)})
        # Convert results to DataFrame
        return pd.DataFrame(csi_results).sort_values(by="CSI", ascending=False) 
    
    
    ## Getting FI for all features and those contributing to 95%
    def feature_importance(self, model, cols_train):
        """
        Computes feature importance for the given model and plots the top features.

        Parameters:
        model : fitted model
            The trained model from which to derive feature importance.
        cols_train : list
            List of feature names used during training.

        Returns:
        feat_imp_df : pd.DataFrame
            DataFrame of features and their respective importance scores.
        selected_features_95 : pd.DataFrame
            Features that contribute to 95% of cumulative importance.
        """
        # Extract feature importance and names
        feature_importance = model.feature_importances_
        feature_names = cols_train
        feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
        feat_imp_df['Cumulative_Importance'] = feat_imp_df['Importance'].cumsum()
        
        # 1. Select features covering 95% of cumulative importance
        selected_features_95 = feat_imp_df[feat_imp_df['Cumulative_Importance'] <= 0.95]
        
        top_f = feat_imp_df.sort_values(by='Importance', ascending=False).head(35)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=top_f['Importance'], 
            y=top_f['Feature'], 
            palette="viridis", 
            ax=ax
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Top Selected Features for Next Model Iteration', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.show() 
        
        return feat_imp_df, selected_features_95
    
    
    def auc_plot(self, model, y, test):
        """
        Plots the ROC curve and computes AUC for the given model and test data.

        Parameters:
        model : fitted model
            The trained model used to predict probabilities.
        y : pd.Series
            Actual outcomes to compare against.
        test : pd.DataFrame
            Features of the test dataset.

        Returns:
        None
        """
        # Compute probabilities and AUC for training data
        train_probabilities = model.predict_proba(test)[:, 1] 
        train_roc_auc = roc_auc_score(y, train_probabilities)
        train_fpr, train_tpr, _ = roc_curve(y, train_probabilities)
        train_gini = 2 * train_roc_auc - 1  
        
        plt.figure(figsize=(10, 8))
        plt.plot(train_fpr, train_tpr, color='green', lw=2, 
                 label=f'Training ROC (AUC = {train_roc_auc:.2f}, Gini = {train_gini:.2f})')   
        
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Classifier (AUC = 0.50)')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=15)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()

        print(f"Gini Coefficient: {train_gini:.4f}")

    
    ## Classification Report;
    def classification_report(self, model, threshold, y_valid, test):
        """
        Generates and prints a classification report including confusion matrix and accuracy score.

        Parameters:
        model : fitted model
            The trained model used to predict probabilities.
        threshold : float
            Probability threshold for classifying predictions as positive.
        y_valid : pd.Series
            Actual outcomes to compare against.
        test : pd.DataFrame
            Features of the validation dataset.

        Returns:
        None
        """
        probabilities = model.predict_proba(test)[:, 1]  
        print(f"\n============ Classification Report for Threshold = {threshold} ============\n")
        # Convert probabilities to binary predictions based on threshold
        predictions = (probabilities >= threshold).astype(int)
        conf_matrix = confusion_matrix(y_valid, predictions)
        print(f"Confusion Matrix:\n{conf_matrix}\n")
        print('    ')
        accuracy = accuracy_score(y_valid, predictions)
        print(f"Accuracy: {accuracy:.4f}\n")

        class_report = classification_report(y_valid, predictions)
        print(f"Classification Report:\n{class_report}")
        print('    ')
        print('    ')    

    def calculate_shap_values(self, df, model):
        """
        Calculates SHAP values for a given dataframe and model, and plots the feature importance.
        
        Parameters:
        df (pd.DataFrame): The input dataframe (without target variable).
        model: A trained machine learning model that supports SHAP (e.g., XGBoost, LightGBM, RandomForest, etc.).
        
        Returns:
        pd.DataFrame: A dataframe containing SHAP values for each feature.
        """
        # Initialize the SHAP explainer
        explainer = shap.Explainer(model, df)
        
        # Compute SHAP values
        shap_values = explainer(df)
        
        # Convert to DataFrame
        shap_df = pd.DataFrame(shap_values.values, columns=df.columns)
        
        # Plot the SHAP summary
        shap.summary_plot(shap_values, df)
        
        return shap_df
    
    
    def chunks_interval_for_reading_query(self, cols, chunk_size):
        """
        Generates intervals for reading columns in chunks.

        Parameters:
        cols : list
            List of column names.
        chunk_size : int
            Size of each chunk.

        Returns:
        all_cols : list of tuples
            List of tuples containing start and end indices for each chunk.
        """
        all_cols = [(i, min(i + 100, len(cols))) for i in range(0, len(cols), chunk_size)]
        return all_cols
    
    
    ### Storing models/pipelines objects as a pickle file;
    def store_pickle(self, model, folder_name, file_name):
        """
        Saves a model object as a pickle file to the specified directory.

        Parameters:
        model : object
            The model or object to be saved.
        folder_name : str
            Directory where the model will be saved.
        file_name : str
            Name of the file to save the model (without extension).

        Returns:
        None
        """
        # Define the folder path where you want to save the model
        save_dir = f"{folder_name}/"  
        os.makedirs(save_dir, exist_ok=True) 
        # Define the full path for the pickle file
        model_filename = os.path.join(save_dir, f"{file_name}.pkl")  
        # Save the model
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model successfully saved to {model_filename}")        
        
    ### Storing models/pipelines objects as a pickle file;
    def read_pickle(self, folder_name, file_name):
        """
        Loads a model object from a pickle file.

        Parameters:
        folder_name : str
            Directory from which the model will be loaded.
        file_name : str
            Name of the file to load the model (without extension).

        Returns:
        loaded_object : object
            The loaded model or object.
        """
        path = os.path.join(folder_name, file_name)
        # Load the model
        with open(path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object

    ## Saving a pickle file(model) into s3 bucket (AWS);
    def store_pickle_AWS(self, model_path, s3_bucket, s3_key): 
        """
        Saves a model pickle file to an AWS S3 bucket.

        Parameters:
        model_path : str
            Path to the model file on the local system to be uploaded.
        s3_bucket : str
            Name of the S3 bucket.
        s3_key : str
            Key under which the model will be stored in the S3 bucket.

        Returns:
        None
        """
        import io
        model = pickle.load(open(model_path, "rb"))
        # Serialize data to pickle format into a BytesIO stream
        pickle_buffer = io.BytesIO()
        pickle.dump(model, pickle_buffer)
        pickle_buffer.seek(0)  # Move to the start of the stream
        # Initialize the S3 client
        s3_client = boto3.client("s3")
        # Upload the pickle file to S3
        s3_client.upload_fileobj(
            Fileobj=pickle_buffer,
            Bucket=s3_bucket,
            Key=s3_key
        )

    ## Reading the saved pickle file from s3 (AWS);
    def read_pickle_AWS(self, s3_bucket, s3_key):
        """
        Loads a model pickle file from an AWS S3 bucket.

        Parameters:
        s3_bucket : str
            Name of the S3 bucket.
        s3_key : str
            Key under which the model is stored in the S3 bucket.

        Returns:
        model : object
            The loaded model or object.
        """
        # Initialize the S3 client
        s3_client = boto3.client("s3")
        # Fetch the pickle file from S3
        pickle_buffer = io.BytesIO()
        s3_client.download_fileobj(Bucket=s3_bucket, Key=s3_key, Fileobj=pickle_buffer)
        # Move to the start of the buffer
        pickle_buffer.seek(0)
        # Deserialize the pickle file
        model = pickle.load(pickle_buffer)

        return model    


    def get_library_versions(self):
        """
        Returns a dictionary of installed libraries along with their respective versions.
        
        Returns:
        dict: A dictionary where keys are library names and values are their versions.
        """
        installed_packages = pkg_resources.working_set
        library_versions = {pkg.project_name: pkg.version for pkg in installed_packages}
        return library_versions    


class model_experimentation:
    def __init__(self, folder_name: str, feature_list: list, model = XGBClassifier(enable_categorical = True)):
        """
        Initializes the model experimentation setup.

        Parameters:
        folder_name : str
            Directory where the output files (models and results) will be saved.
        feature_list : list
            List of features to be considered in the model training.
        model : XGBClassifier, optional
            Instance of XGBClassifier to be used for training (default is initialized with categorical support).
        """
        os.makedirs(folder_name, exist_ok=True)
        self.folder_name = folder_name
        self.model = model
        self.helper_object = evaluation()
        self.probs = []
        self.feature_list = feature_list

    def model_training(self, cols_to_remove_feat: list, target: str, train):
        """
        Trains the XGB model by removing specified features.

        Parameters:
        cols_to_remove_feat : list
            List of column names to be removed from the training dataset.
        target : str
            The name of the target variable in the training dataset.
        train : DataFrame
            The training dataset containing features and the target variable.
        
        Returns:
        None
        """
        ## removing the columns
        X, y = train.drop(cols_to_remove_feat, axis=1), train[[target]]
        self.model.fit(X, y)
        print('Finished Training!!')

        ## Generating the probabilities for each decile;
        y_pred = self.model.predict_proba(X)[:, 1]     
        self.probs = self.helper_object.decile_probs(y_pred, 10)  ### Hard coded the bins as 10, can be changed
        self.probs = pd.DataFrame(self.probs, columns=['probs'])    

    def saving_requirements(self):
        """
        Saves the trained model, probability deciles, and feature importance.

        Parameters:
        None
        
        Returns:
        None
        """
        ## Saving the models;
        self.helper_object.store_pickle(self.model, self.folder_name, 'model_file')  
        ## Saving the deciles;
        self.probs.to_csv(f'{self.folder_name}/train_probs.csv', index=False)
        ## Saving the feature importances;
        feat_imp_df, selected_features_95 = self.helper_object.feature_importance(self.model, self.feature_list)
        feat_imp_df.to_csv(f'{self.folder_name}/feature_importance.csv', index=False)

    def randomized_search_xgboost_with_validation(self, X_train, y_train, X_test, y_test, param_dist, n_iter=50, random_state=42):
        """
        Searches for the best hyperparameters for XGBoost using randomized search with validation.

        Parameters:
        X_train : DataFrame
            Training features.
        y_train : Series/DataFrame
            Training target variable.
        X_test : DataFrame
            Testing features.
        y_test : Series/DataFrame
            Testing target variable.
        param_dist : dict
            Dictionary where keys are parameter names and values are lists of parameter settings to try.
        n_iter : int, optional
            Number of iterations for parameter sampling (default is 50).
        random_state : int, optional
            Seed for random number generator (default is 42).

        Returns:
        auc_results_df : DataFrame
            DataFrame containing parameter combinations and their corresponding validation AUC scores.
        """
        
        # Create the XGBoost classifier
        xgb_classifier = XGBClassifier()  ## eval_metric='logloss', enable_categorical=True, scale_pos_weight = 
    
        # List to hold the results
        auc_results = []
    
        # Set to keep track of unique parameter tuples
        unique_combinations = set()
    
        # Random seed for reproducibility
        np.random.seed(random_state)
    
        # Perform Randomized Search CV
        for i in tqdm(range(n_iter)):
            # Sample hyperparameters
            params = {key: np.random.choice(value) for key, value in param_dist.items()}
            param_tuple = tuple(sorted(params.items()))  # Convert to a hashable tuple
    
            # Check for uniqueness
            if param_tuple not in unique_combinations:
                unique_combinations.add(param_tuple)
    
                # Fit the model on training data
                xgb_classifier.set_params(**params)
                xgb_classifier.fit(X_train, y_train)
    
                # Get the AUC score on the validation set
                train_auc = roc_auc_score(y_train, xgb_classifier.predict_proba(X_train)[:, 1])
                val_auc = roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:, 1])
    
                # Log the hyperparameters and AUC score
                auc_results.append((params, val_auc, train_auc))
                
        # Convert logged results to a DataFrame
        auc_results_df = pd.DataFrame(auc_results, columns=['Parameters', 'Test AUC', 'Train AUC'])
        
        return auc_results_df


