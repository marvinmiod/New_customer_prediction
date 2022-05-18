# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:26:04 2022

The criteria of the project are as follows:
1) Develop a deep learning model using TensorFlow which only comprises of Dense, Dropout, and Batch Normalization layers.
2) The accuracy of the model must be more than 80% with F1 more than 80%.
3) Display the training loss and accuracy on TensorBoard
4) Create modules (classes) for repeated functions to ease your training and testing process


@author: Marvin
"""



import pandas as pd
import numpy as np
import datetime
import os
import pickle
import missingno as msno
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential #model is only for Sequential Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization # to add after hidden layer
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.models import load_model


#%% Saved path

# save the model and save the log
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'predict_new_customer.h5')
# path where the logfile for tensorboard call back
LOG_PATH = os.path.join(os.getcwd(),'Log_new_customer')
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
MINMAX_SCALER_PATH = os.path.join(os.getcwd(), 'minmax_scaler.pkl')
ONE_HOT_SCALER_PATH = os.path.join(os.getcwd(), 'one_hot_scaler.pkl')
DATASET = os.path.join(os.getcwd(), 'dataset', 'new_customers.csv')



#%%



#%%# EDA

# Step 1) Load data
df = pd.read_csv(DATASET)
# create dummy data to work with
dummy_df = df.copy()

# Step 2) Data Inspection
df.head()
df.describe().T # check for outlier
df.dtypes
df.info()
# visualise the data
df.boxplot()
# To check/visualise if there is any missing data
df.isna().sum()

# to visualize the missing numbers in the dataframe
msno.matrix(df)
msno.bar(df)


#%% Step 3) Data Cleaning

# map text data to number and remain missing data as NaN
dummy_df['Gender'] = dummy_df['Gender'].map({'Male':1,'Female':0})
dummy_df['Ever_Married'] = dummy_df['Ever_Married'].map({'Yes':1,'No':0})
dummy_df['Graduated'] = dummy_df['Graduated'].map({'Yes':1,'No':0})

# map spending_score text to numbers
dummy_df['Spending_Score'] = dummy_df['Spending_Score'].map({'Low':0,
                                                             'Average':1,
                                                             'High':2})
label_enc = LabelEncoder()
#convert Profession data from text label to numbers using label encoder
# NaN values in profession is converted to num 9 using label encoder
dummy_df['Profession'] = dummy_df['Profession'].astype(str)
dummy_df['Profession'] = label_enc.fit_transform(dummy_df['Profession'])

#convert Var_1 data from text label to numbers using label encoder
# NaN values in Var_1 is converted to num 7 using label encoder
dummy_df['Var_1'] = dummy_df['Var_1'].astype(str)
dummy_df['Var_1'] = label_enc.fit_transform(dummy_df['Var_1'])

# Segmentation text data convert to numbers
# segmentation is NaN because this we need to predict which segment based on data
#dummy_df['Segmentation'] = label_enc.fit_transform(dummy_df['Segmentation'])

#drop ID and segmentation from the data as the x_features in the training model
# has the ID and segmentation removed
dummy_df = dummy_df.drop(columns=['ID','Segmentation']) 

#%% Step 3b) Impute missing (NaN) values using Iterative imputer


# initialize imputer with iterative imputer and fit dummy_df into it.
imputer = IterativeImputer()
dummy_df_iterative = imputer.fit_transform(dummy_df)

#convert to data frame then only can view back whether got NaN or not
dummy_df_iterative = pd.DataFrame(dummy_df_iterative)


# check back the dummy data that has been imputed for any NaN values
dummy_df_iterative.isna().sum() # All zero NaN
dummy_df_iterative.describe().T # all 8068 count data in each column
msno.matrix(dummy_df_iterative) # visualise dataframe that has been imputed



#%% Loading of Settings or Model
#minmax_scaler = pickle.load(open(MINMAX_SCALER_PATH, 'rb'))
#one_hot_scaler = pickle.load(open(ONE_HOT_SCALER_PATH, 'rb'))



# if using Deep Learning to load model
model = load_model(MODEL_SAVE_PATH)
model.summary()


#mm_scaler = MinMaxScaler()
#x_features_scaled = mm_scaler.fit_transform(x_features)
#pickle.dump(mm_scaler, open('minmax_scaler.pkl', 'wb'))

# fit the x_feature (9 column dataset) into min max scaler
mm_scaler = MinMaxScaler()
x_features_scaled = mm_scaler.fit_transform(dummy_df_iterative)

# predict the test data into the model
outcome = model.predict(x_features_scaled)
print(outcome)

# need to inverse the value for the outcome to predict segmentation of customer

