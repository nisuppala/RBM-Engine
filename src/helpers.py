# AMRUTA OXAI V2
# Copyright Amruta Inc. 2021
# Author: Dishit Kotecha
# Disclaimer: Code subject to change any time. Amruta Inc. is not responsible for any loss/damage caused by use of this code
# Purpose: Be able to load any predictive supervised learning model
# and dataset to understand what features affect your output the most
# using Explainable AI methods

import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier, DMatrix
import xgboost as xgb
# from sklearn.tree import export_graphviz
# from graphviz import Source
import shap
# from dtreeviz.trees import *
import base64
import operator
from sklearn.impute import SimpleImputer
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

@st.cache(hash_funcs={dict:id}, allow_output_mutation=True)
def load_data(data, separator):
    df = pd.read_csv(data, sep=separator)
    return df


@st.cache
def df_summary(df):
    return df.describe()

def first_admit_function(df, patient_id,admit_date,readmit_flag):
    df[readmit_flag]= df[readmit_flag].fillna(0)
    df = df.sort_values(by = [patient_id, admit_date], ascending = [True,True])
    df['First_Readmit'] = df.groupby([patient_id])[readmit_flag].shift(-1)
    df['First_Readmit']= df['First_Readmit'].fillna(0)  
    df = df[df[readmit_flag] != 1]
    df = df.drop([readmit_flag],axis = 1)
    return df

def weight_by_freq(inp_df,list_var, hospital_code):
    return_df = inp_df.copy()     # create a copy of the dataframe
    column_list = list_var
    temp_df = return_df.loc[:, column_list]     # filter the df with selected dataframe
    var_list = temp_df.values.tolist()          # Get all the values in the dataframe
    flat_list = [item for sublist in var_list for item in sublist]  # create a flat list from list in a list 
    flat_list = [string for string in flat_list if string != " "]   # remove all the empty strings 
    my_dict = Counter(flat_list) # use the counter to get the count of the values
    for value in my_dict.values():  
        value = int(value)
        value = (temp_df.shape[0]*temp_df.shape[1])/(value*100)  # get the freq value by dividing it by the no of obs and no of columns and then inverting it 
    for i in column_list:
        return_df[i]= return_df[i].map(my_dict)  # search for the values in the dictionary and repalce it
    if hospital_code == 'Diagnostic':
        column_name = 'Weight_Dx_Code'
    elif hospital_code == 'Procedural':
        column_name = 'Weight_Proc_Code'
    inp_df[column_name] = return_df[column_list].sum(axis=1)
    return inp_df

def weight_by_count(input_df, p_id, e_id, list_var, hospital_code):
    df = pd.DataFrame()    # initialise empty dataframe
    df['Patient_id'] = input_df[p_id]  #Get the Patient id from the input_df
    df['Encounter_id'] = input_df[e_id]    #get the Encounter id from the input_df
    temp_df = input_df.loc[:, list_var] #get all the columns that starts with the var & save it as temp_df 
    temp_df_columns = temp_df.columns.values.tolist() # get the column names of the temp_df
    df = pd.concat([df, temp_df], axis=1) # concat the columns
    melt_df=df.melt(['Patient_id', 'Encounter_id'], value_vars = temp_df_columns) # melt the dataframe
    melt_df = melt_df.drop(['variable'],axis =1) # drop the variable 
    melt_df = melt_df.replace(r'^\s*$', np.NaN, regex=True) # replace all teh empty strings with NAN
    group_df = melt_df.groupby(['Patient_id', 'Encounter_id'], dropna=True).count().reset_index() #Group by Patient_id and Encounter_id
    if hospital_code == 'Diagnostic':
        column_name = 'Count_Dx_Code'
    elif hospital_code == 'Procedural':
        column_name = 'Count_Proc_Code'
    input_df[column_name] = group_df['value'].to_list()
    return input_df #Return the result 

def null_impute(df, columns, imputer_method):
    null_df = df.copy()
    for i in range(len(columns)):
        if imputer_method[i] != 'Iterative Imputer' and imputer_method[i] != 'Please select imputer' and imputer_method[i] != 0:
            impute = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=imputer_method[i])
            null_df[columns[i]] = impute.fit_transform(null_df[columns[i]].values.reshape(-1, 1))
        elif imputer_method[i] == 'Iterative Imputer':
            nn_df = null_df[null_df.columns[~null_df.isnull().any()]]
            nn_df = nn_df.select_dtypes([np.number])
            nn_df[columns[i]] = null_df[columns[i]]
            impute_iterative = IterativeImputer(skip_complete=True)
            nn_imputed_df = pd.DataFrame(impute_iterative.fit_transform(nn_df))
            nn_imputed_df.columns = nn_df.columns
            null_df[columns[i]] = nn_imputed_df[columns[i]]
    return null_df


def label_encode(df, columns):
    encoded_df = df.copy()
    le = LabelEncoder()
    for col in columns:
        encoded_df[col] = encoded_df[col].astype('str')
        encoded_df[col] = le.fit_transform(encoded_df[col])
        st.write(col)
        st.write(pd.DataFrame(zip(le.classes_, le.transform(le.classes_)), columns=['value', 'encoding']))
    return encoded_df

def standard_scale(df, columns):
    scaled_df = df.copy()
    scaler = StandardScaler()
    scaler.fit(df[columns])
    scaled_df[columns] = scaler.transform(df[columns])
    return scaled_df

@st.cache(hash_funcs={pd.core.frame.DataFrame: id})
def process_data(cols_to_drop, cols_to_encode, df, encode_method):
    if cols_to_drop != None:
        df = df.drop(cols_to_drop, axis=1)

    if cols_to_encode != None:
        if encode_method == 'Label Encoder':
            df = label_encode(df, cols_to_encode)
        elif encode_method == 'One Hot Encoder':
            df = pd.get_dummies(df, columns = cols_to_encode)

    return df

def filter_df(df, op, column, compare_val):
    operators = {
        '>':operator.gt,
        '<':operator.lt,
        '=':operator.eq
    }

    if op in operators:
        return df[operators[op](df[column], compare_val)]
    elif op == 'contains':
        return df[df[column] == compare_val]
    elif op == 'does not contain':
        return df[df[column] != compare_val]

@st.cache
def calc_corr(df):
    return df.corr()

def render_heatmap(df):
    """displays correlation matrix in heatmap format using Seaborn"""

    cor = calc_corr(df)
    if cor.shape[0] < 15:
        plt.figure(figsize=(30,30))
        sns.heatmap(cor, cmap='YlGnBu', annot=True, annot_kws={"size": 12})
        st.pyplot(bbox_inches='tight')
    else:
        st.write(cor)



#########################################################
################### EXPLAINABILITY ######################
#########################################################


###### LIME ######
@st.cache(suppress_st_warning=True)
def lime_explainability(model, xtest, ytest, yscore, pred_var):
    categorical_features = np.argwhere(np.array([len(set(xtest.values[:,x]))
                                                 for x in range(xtest.values.shape[1])]) <= 20).flatten()
    explainer = lime.lime_tabular.LimeTabularExplainer(xtest.values,
                                                       feature_names = xtest.columns.values.tolist(),
                                                       class_names = [pred_var],
                                                       categorical_features=categorical_features,
                                                       verbose = True,
                                                       mode = 'classification')

    return explainer


# def plot_tree_model(model, x, y, ypred, pred_var, model_type):

#     if (model_type == 'XGBoost Regressor') or (model_type == 'XGBoost Classifier'):
#         num_boosters = model.get_booster().best_iteration + 1
#         st.write("Number of boosters: %d"%(num_boosters))
#         ntree=st.number_input('Select the desired tree for visualization'
#                                                     , min_value=int(min(range(model.get_booster().best_iteration)))
#                                                     , max_value=int(max(range(model.get_booster().best_iteration+1))),
#                                                     value = 0)

#         tree=xgb.to_graphviz(model,num_trees=ntree)
#         st.graphviz_chart(tree)
#     elif ('Decision Tree' in model_type):
#         tree_load_state = st.text('Loading Tree Visualization...')

#         individual = st.number_input('Select desired observation for explanation',
#                                      min_value=min(range(len(x))),
#                                      max_value=max(range(len(x))),
#                                      value=0)
#         fancy = st.checkbox('Show histograms and scatterplots at every node?')
#         predicted = np.array(ypred)[individual]
#         actual = np.array(y)[individual]
#         st.write('Actual = ', actual)
#         st.write('Predicted = ',predicted)
#         X = x.iloc[individual, :]
#         if 'Classifier' in model_type:
#             class_names = list(set(y.values))
#         else:
#             class_names = None
#         viz = dtreeviz(model, x, y,
#                        target_name=pred_var,
#                        feature_names=x.columns,
#                        orientation='LR',X=X,
#                        class_names=class_names,
#                        fancy=fancy)
#         render_svg(viz.svg())
#         tree_load_state.text('Loading Tree Visualization... done!')
#         #st.graphviz_chart(viz)
#         #render_svg(viz)

#     elif ('Random Forest' in model_type):

#         estimators = model.estimators_
#         num_estimators = len(estimators)
#         st.write("Number of trees/estimators: %d"%(num_estimators))

#         estimator_index = st.number_input('Select which estimator from your RF model to view',
#                                      min_value=min(range(num_estimators)),
#                                      max_value=max(range(num_estimators)),
#                                      value=0)

#         estimator = estimators[estimator_index]

#         individual = st.number_input('Select desired observation for explanation',
#                                      min_value=min(range(len(x))),
#                                      max_value=max(range(len(x))),
#                                      value=0)
#         fancy = st.checkbox('Show histograms and scatterplots at every node?')
#         tree_load_state = st.text('Loading Tree Visualization...')
#         predicted = np.array(ypred)[individual]
#         actual = np.array(y)[individual]
#         st.write('Actual = ', actual)
#         st.write('Predicted = ',predicted)
#         X = x.iloc[individual, :]

#         if 'Classifier' in model_type:
#             class_names = list(set(y.values))
#         else:
#             class_names = None
#         viz = dtreeviz(estimator, x, y,
#                        target_name=pred_var,
#                        feature_names=x.columns,
#                        orientation='LR',X=X,
#                        class_names=class_names,
#                        fancy=fancy)

#         render_svg(viz.svg())
#         tree_load_state.text('Loading Tree Visualization... done!')

#     else:
#         st.error('Tree Visualization only available for tree based models')

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="3000" height="3000"/>' % b64
    st.write(html, unsafe_allow_html=True)

