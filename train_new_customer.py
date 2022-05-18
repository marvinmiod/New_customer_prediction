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


#%%

# save the model and save the log
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'predict_new_customer.h5')
# path where the logfile for tensorboard call back
LOG_PATH = os.path.join(os.getcwd(),'Log_new_customer')
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
DATASET = os.path.join(os.getcwd(), 'dataset', 'train.csv')

#%%
# Function Definition

def create_model(output_node, input_data_shape, 
                 nb_nodes, activation='relu'):
    """
    This Function creates a model with 2 hidden layers
    Last layers of the model comprises of softmax activation function.
    ----------
    output_node : Int
        Contains the output number (vertical).
    input_data_shape : Array
        Contains the shape of the input.
    nb_nodes : Int, optional
        DESCRIPTION. The default is 32.
    activation : String, optional
        DESCRIPTION. The default is 'relu'.

    Returns
    -------
    Created Model.

    """
    input_1 = Input(shape=input_data_shape)
    #flatten_layer = Flatten()(input_1) #must import Flatten on the top
    #hidden_1 = Dense(512, activation='sigmoid', name='hidden_layer_1')(flatten_layer)
    # use activation=relu (for better result) to avoid exploding and vanishing gradients
    hidden_1 = Dense(nb_nodes, activation=activation, 
                     name='hidden_layer_1')(input_1)
    
    batch_norm_1 = BatchNormalization()(hidden_1)
    dropout_layer_1 = Dropout(0.2)(batch_norm_1) # dropout 0.2 means 20% from the layer
    #hidden_2 = Dense(512, activation='sigmoid', name='hidden_layer_2')(dropout_layer_1)
    # use activation=relu (for better result) to avoid exploding and vanishing gradients
    hidden_2 = Dense(nb_nodes, activation=activation, 
                     name='hidden_layer_2')(dropout_layer_1)
    
    batch_norm_2 = BatchNormalization()(hidden_2)
    dropout_layer_2 = Dropout(0.2)(batch_norm_2)
    output_1 = Dense(output_node, activation='softmax', 
                     name='output_layer')(dropout_layer_2)
    
    model = Model(inputs=[input_1], outputs=[output_1])
    model.summary()
    plot_model(model)

    return model

def create_model_dl(output_node, input_data_shape, 
                    nb_nodes, activation='relu'):
    """
    This Function creates a model with 3 hidden layers
    Last layers of the model comprises of softmax activation function.
    ----------
    output_node : Int
        Contains the output number (vertical).
    input_data_shape : Array
        Contains the shape of the input.
    nb_nodes : Int, optional
        DESCRIPTION. The default is 32.
    activation : String, optional
        DESCRIPTION. The default is 'relu'.

    Returns
    -------
    Created Model.

    """
    
    input_1 = Input(shape=input_data_shape)
#    flatten_layer = Flatten()(input_1) #must import Flatten on the top
    hidden_1 = Dense(nb_nodes, activation=activation, 
                     name='hidden_layer_1')(input_1)
    # use activation=relu (for better result) to avoid exploding and vanishing gradients
    #hidden_1 = Dense(nb_nodes, activation=activation, 
    #                 name='hidden_layer_1')(input_1)
    
    batch_norm_1 = BatchNormalization()(hidden_1)
    dropout_layer_1 = Dropout(0.2)(batch_norm_1) # dropout 0.2 means 20% from the layer
    #hidden_2 = Dense(512, activation='sigmoid', name='hidden_layer_2')(dropout_layer_1)
    # use activation=relu (for better result) to avoid exploding and vanishing gradients
    hidden_2 = Dense(nb_nodes, activation=activation, 
                     name='hidden_layer_2')(dropout_layer_1)    
    batch_norm_2 = BatchNormalization()(hidden_2)
    dropout_layer_2 = Dropout(0.2)(batch_norm_2)
    
    # add additional hidden layer below
    hidden_3 = Dense(nb_nodes, activation=activation, 
                     name='hidden_layer_3')(dropout_layer_2)
    batch_norm_3 = BatchNormalization()(hidden_3)
    dropout_layer_3 = Dropout(0.2)(batch_norm_3)
    
    #output
    output_1 = Dense(output_node, activation='softmax', 
                     name='output_layer')(dropout_layer_3)
    
    model = Model(inputs=[input_1], outputs=[output_1])
    model.summary()
    plot_model(model)

    return model


def training_history(hist):
    
    """
    To visualise the training and validation graph using matplotlib
    ----------
    hist : TYPE
        DESCRIPTION.

    Returns
    -------
    training and validation runs and history of runs.
    """

    keys = [i for i in hist.history.keys()]
    
    training_loss = hist.history[keys[0]] #this is cross_entrophy loss
    training_metric = hist.history[keys[1]] #accuracy of the model
    validation_loss = hist.history[keys[2]]
    validation_metric = hist.history[keys[3]]
   
    # step 5a) model evaluation to evaluate training loss
    plt.figure()
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('Training {} and validation {}'.format(keys[0], keys[0]))
    plt.xlabel('epoch')
    plt.ylabel(keys[0])
    plt.legend(['training loss', 'validation loss'])
    plt.show
  
    # step 5b) model evaluation to evaluate training accuracy
    plt.figure()
    plt.plot(training_metric)
    plt.plot(validation_metric)
    plt.title('Training {} and validation {}'.format(keys[1], keys[1]))
    plt.xlabel('epoch')
    plt.ylabel(keys[1])
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.show


def report_metrics(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    # to print accuracy score its 56% and its bad, like tossing a coin
    print('Accuracy score is', accuracy_score(y_true, y_pred))    
   
def report_generation(x_test, y_test):
    """
    To view report for accuracy, f1-report
    ----------
    x_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    # this one print the accuracy numbers in the console
    pred_x = model.predict(x_test)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(pred_x, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)
    print(cr)
    
    # Code for Confusion Matrix correlation graph
    #labels = [ '0', '1', '2' ,'3', '4', '5', '6' ,'7', '8', '9'] # manual way
    labels = [str(i) for i in range(10)] # using list comprehension
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
    #                              display_labels=np.unique(y_true))
    
    # this one is to removed the display_labels with unique(y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    disp.plot(cmap=plt.cm.Blues)
    plt.show()




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
dummy_df['Segmentation'] = label_enc.fit_transform(dummy_df['Segmentation'])


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


#%% Step 4) Check if there is strong correlation for Features selection 
# outcome= no strong correlation exist

column_names = ['ID','Gender','Ever_Married','Age','Graduated',
                'Profession','Work_Experience','Spending_Score',
                'Family_Size','Var_1','Segmentation']

dummy_df_iterative.columns = column_names
cor = dummy_df_iterative.corr()
plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True, cmap=plt.cm.Reds)
plt.show() 




#%% # Step 5) Data Pre-processing - using MinMaxScaler
#identify x,y train and test data

# to select few features
#x1 = dummy_df_iterative['Profession'] # 0.22 
#x2 = dummy_df_iterative['Work_Experience'] # 0.
#x3 = dummy_df_iterative['Family_Size'] # 0.
#x4 = dummy_df_iterative['Gender'] # 0.
#x = [x1,x2,x3,x4]
#x_features = np.array(x).T 

# uncomment below to take all columns as features and drop segmentation

column_names = ['ID','Gender','Ever_Married','Age','Graduated',
                'Profession','Work_Experience','Spending_Score',
                'Family_Size','Var_1','Segmentation']



# to drop ID and segmentation columns from x_features
x_features = dummy_df_iterative.drop(columns=['ID','Segmentation']) 

#drop ID, var_1 and segmentation from x_features (not required)
#x_features = dummy_df_iterative.drop(columns=['ID','Var_1','Segmentation']) 


# Scale the x_feature using MinMax Scaler so that value is within range of 0-1
mm_scaler = MinMaxScaler()
x_features_scaled = mm_scaler.fit_transform(x_features)
pickle.dump(mm_scaler, open('minmax_scaler.pkl', 'wb'))


# y target is column "segmentation" and perform one hot encoder
y_target = dummy_df_iterative['Segmentation']
one_hot = OneHotEncoder(sparse=False)
y_one_hot = one_hot.fit_transform(np.expand_dims(y_target, axis=-1))

# this is to dump the one_hot_encoder file
pickle.dump(one_hot, open('one_hot_scaler.pkl', 'wb')) 

# Split test and train data
x_train, x_test, y_train, y_test = train_test_split(x_features_scaled, 
                                                    y_one_hot, 
                                                    test_size=0.3)



#%% Step 6a) Functional API Model compilation

input_data_shape = x_features.shape[1] # auto-get x_features input shape
output_node = y_one_hot.shape[1] # auto-get y_target_encoded output shape
nb_nodes = 64
#create_model(output_node, input_data_shape, nb_nodes)
model = create_model_dl(output_node, input_data_shape, nb_nodes) 
plot_model(model)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics='accuracy')





#%%  Step 8)  Model training
# use Tensorboard callback instead of EarlyStopping
tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

hist = model.fit(x_train, y_train, epochs=50, 
                 validation_data=(x_test, y_test),
                 callbacks=[tensorboard_callback, early_stopping_callback])


print(hist.history.keys())
                 
# Step 9) Visualize the training losses using matplotlib
training_history(hist)

# to view the tensorboard
# tensorboard --logdir "<path of log files>"


#%% # Step 10) Discussion / Summary report 
# Model analysis    

report_generation(x_test, y_test)
# The accuracy of the model is between 49% - 52% using 3 hidden layers and 128 nodes
# The model accuracy cannot be improve further even after adding 6 hidden layer
# and also reduced or added nodes from 16 to 1024 nodes.
# Suggestion 1: 
# to perform feature selection to reduce number of input into the training model

# suggestion 2: 
# to use Machine Learning pipeline to perform feature selection

#%% Step 11) Save Model
model.save(MODEL_SAVE_PATH)


