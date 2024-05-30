
#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit.components.v1 as components
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
from streamlit_extras.stoggle import stoggle
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#----------------------------------------
import os
import time
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from random import randint
#----------------------------------------
import json
import holidays
import base64
import itertools
import codecs
from datetime import datetime, timedelta, date
#from __future__ import division
#----------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#----------------------------------------
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyoff
import altair as alt
#import dabl
#----------------------------------------
import sweetviz as sv
import pygwalker as pyg
#----------------------------------------
# Model Building
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
#
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
#import optuna.integration.lightgbm as lgb
from sklearn.metrics import classification_report,confusion_matrix
#----------------------------------------
# Model Performance
import shap
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, chi2, VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
#----------------------------------------
# Model Validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
#----------------------------------------
#from pycaret.classification import setup, compare_models, pull, save_model, evaluate_model
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="Classification App",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="collapsed")
#----------------------------------------
st.title(f""":rainbow[ML | Classification App | v2.0]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
stats_expander = st.expander("**Knowledge**", expanded=False)
with stats_expander: 
      st.info('''
        **Classification**
                       
        - A supervised learning method which generates the probability of the target variables based on selected features.
        ''')
st.divider()
#---------------------------------------------------------------------------------------------------------------------------------
### Feature Import
#---------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header("Contents", divider='blue')
st.sidebar.info('Please choose from the following options and follow the instructions to start the application.', icon="ℹ️")
data_source = st.sidebar.radio("**:blue[Select Data Source]**", ["Local Machine", "Server Path"])
#---------------------------------------------------------------------------------------------------------------------------------

if data_source == "Local Machine" :
    
    file1 = st.sidebar.file_uploader("**:blue[Choose a file]**",
                                type=["xlsx","csv"],
                                accept_multiple_files=True,
                                key=0)
    #st.sidebar.divider()       
    if file1 is not None:
        df = pd.DataFrame()
        for file1 in file1:
            df = pd.read_csv(file1)
            #st.subheader("Preview of the Input Dataset :")
            #st.write(df.head(3))

            # Dataset preview
            if st.sidebar.checkbox("**Preview Dataset**"):
                number = st.sidebar.slider("**Select No of Rows**",0,df.shape[0],3,5)
                st.subheader("**Preview of the Input Dataset :**",divider='blue')
                st.write(df.head(number))
            #st.sidebar.divider()   

            if st.sidebar.checkbox("**🗑️:blue[Feature Drop]**"):
                feature_to_drop = st.sidebar.selectbox("**Select Feature to Drop**", df.columns)
                #df_dropped = df.drop(columns=[feature_to_drop])
                if feature_to_drop:
                    #col1, col2, col3 = st.columns([1, 0.5, 1])
                    if st.sidebar.button("Apply", key="delete"):
                        st.session_state.delete_features = True
                        st.session_state.df = df.drop(feature_to_drop, axis=1)
            st.sidebar.divider() 
            
            st.sidebar.subheader("Variables", divider='blue')
            target_variable = st.sidebar.selectbox("**:blue[Target (Dependent) Variable]**", df.columns)
            #feature_columns = st.sidebar.multiselect("**2.2 Independent Variables**", df.columns)

            st.sidebar.subheader("Algorithm",divider='blue')          
            classifiers = ['logistic_regression',
                            'decision_tree_classification', 
                            'random_forest_classification', 
                            'gradient_boosting',
                            'xtreme_gradient_boosting']
            algorithms = st.sidebar.selectbox("**:blue[Choose an algorithm for predictions]**", options=classifiers)

            classifier_type = ['binary','multi_class']
            classifier_clv = st.sidebar.selectbox("**:blue[Choose the type of classifiers]**", options=classifier_type)   

            if st.sidebar.button(":blue[Proceed]"):

#---------------------------------------------------------------------------------------------------------------------------------
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["**Information**",
                                                                            "**Visualizations**",
                                                                            "**Cleaning**",
                                                                            "**Transformation**",
                                                                            "**Selection**",
                                                                            "**Development & Tuning**",
                                                                            "**Performance**",
                                                                            "**Validation**",
                                                                            "**Cross-Check**"])
#---------------------------------------------------------------------------------------------------------------------------------
### Informations
#---------------------------------------------------------------------------------------------------------------------------------

                with tab1:
                    st.subheader("**Characteristics**",divider='blue')

                    col1, col2, col3, col4, col5, col6 = st.columns(6)

                    col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
                    col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
                    col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                    col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                    st.divider()           

                    stats_expander = st.expander("**Exploratory Data Analysis (EDA)**", expanded=False)
                    with stats_expander:        
                        #pr = df.profile_report()
                        #st_profile_report(pr)
                        st.table(df.head()) 

#---------------------------------------------------------------------------------------------------------------------------------
### Visualizations
#---------------------------------------------------------------------------------------------------------------------------------

                with tab2: 

                    st.subheader("Visualization | Playground",divider='blue')
        
                    pyg_html = pyg.to_html(df)
                    components.html(pyg_html, height=1000, scrolling=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Feature Cleaning
#---------------------------------------------------------------------------------------------------------------------------------

                with tab3:
                
                    st.subheader("Missing Values Check & Treatment",divider='blue')
                    col1, col2 = st.columns((0.2,0.8))

                    with col1:
                        @st.cache_data(ttl="2h")
                        def check_missing_values(data):
                            missing_values = data.isnull().sum()
                            missing_values = missing_values[missing_values > 0]
                            return missing_values 
                        missing_values = check_missing_values(df)

                        if missing_values.empty:
                            st.success("**No missing values found!**")
                        else:
                            st.warning("**Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing_values)

                            with col2:        
                                #treatment_option = st.selectbox("**Select a treatment option**:", ["Impute with Mean","Drop Missing Values", ])
        
                                # Perform treatment based on user selection
                                #if treatment_option == "Drop Missing Values":
                                    #df = df.dropna()
                                    #st.success("Missing values dropped. Preview of the cleaned dataset:")
                                    #st.table(df.head())
            
                                #elif treatment_option == "Impute with Mean":
                                    #df = df.fillna(df.mean())
                                    #st.success("Missing values imputed with mean. Preview of the imputed dataset:")
                                    #st.table(df.head())
                                 
                                # Function to handle missing values for numerical variables
                                @st.cache_data(ttl="2h")
                                def handle_numerical_missing_values(data, numerical_strategy):
                                    imputer = SimpleImputer(strategy=numerical_strategy)
                                    numerical_features = data.select_dtypes(include=['number']).columns
                                    data[numerical_features] = imputer.fit_transform(data[numerical_features])
                                    return data

                                # Function to handle missing values for categorical variables
                                @st.cache_data(ttl="2h")
                                def handle_categorical_missing_values(data, categorical_strategy):
                                    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
                                    categorical_features = data.select_dtypes(exclude=['number']).columns
                                    data[categorical_features] = imputer.fit_transform(data[categorical_features])
                                    return data            

                                numerical_strategies = ['mean', 'median', 'most_frequent']
                                categorical_strategies = ['constant','most_frequent']
                                st.write("**Missing Values Treatment:**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    selected_numerical_strategy = st.selectbox("**Select a strategy for treatment : Numerical variables**", numerical_strategies)
                                with col2:
                                    selected_categorical_strategy = st.selectbox("**Select a strategy for treatment : Categorical variables**", categorical_strategies)  
                                
                                #if st.button("**Apply Missing Values Treatment**"):
                                cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                st.table(cleaned_df.head(2))

                                # Download link for treated data
                                st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                    #with col2:

                    st.subheader("Duplicate Values Check",divider='blue') 
                    if st.checkbox("Show Duplicate Values"):
                        if missing_values.empty:
                            st.table(df[df.duplicated()].head(2))
                        else:
                            st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                    #with col4:

                    #x_column = st.selectbox("Select x-axis column:", options = df.columns.tolist()[0:], index = 0)
                    #y_column = st.selectbox("Select y-axis column:", options = df.columns.tolist()[0:], index = 1)
                    #chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=x_column,y=y_column)
                    #st.altair_chart(chart, theme=None, use_container_width=True)  

                    st.subheader("Outliers Check & Treatment",divider='blue')
                    @st.cache_data(ttl="2h")
                    def check_outliers(data):
                        # Assuming we're checking for outliers in numerical columns
                        numerical_columns = data.select_dtypes(include=[np.number]).columns
                        outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])

                        for column in numerical_columns:
                            Q1 = data[column].quantile(0.25)
                            Q3 = data[column].quantile(0.75)
                            IQR = Q3 - Q1

                            # Define a threshold for outliers
                            threshold = 1.5

                            # Find indices of outliers
                            outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))

                            # Count the number of outliers
                            num_outliers = outliers_indices.sum()
                            outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)

                        return outliers
                
                    if missing_values.empty:
                        df = df.copy()
                    else:
                        df = cleaned_df.copy()

                    col1, col2 = st.columns((0.2,0.8))

                    with col1:
                        # Check for outliers
                        outliers = check_outliers(df)

                        # Display results
                    if outliers.empty:
                        st.success("No outliers found!")
                    else:
                        st.warning("**Outliers found!**")
                        st.write("**Number of outliers:")
                        st.table(outliers)
                    
                    with col2:
                        # Treatment options
                        treatment_option = st.selectbox("**Select a treatment option:**", ["Cap Outliers","Drop Outliers", ])

                        # Perform treatment based on user selection
                        if treatment_option == "Drop Outliers":
                            df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                            st.success("Outliers dropped. Preview of the cleaned dataset:")
                            st.write(df.head())

                        elif treatment_option == "Cap Outliers":
                            df = df.copy()
                            for column in outliers['Column'].unique():
                                Q1 = df[column].quantile(0.25)
                                Q3 = df[column].quantile(0.75)
                                IQR = Q3 - Q1
                                threshold = 1.5

                                # Cap outliers
                                df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])

                                st.success("Outliers capped. Preview of the capped dataset:")
                                st.write(df.head())

#---------------------------------------------------------------------------------------------------------------------------------
### Feature Encoding
#---------------------------------------------------------------------------------------------------------------------------------
                #for feature in cleaned_df.columns: 
                    #if cleaned_df[feature].dtype == 'object': 
                        #print('\n')
                        #print('feature:',feature)
                        #print(pd.Categorical(cleaned_df[feature].unique()))
                        #print(pd.Categorical(cleaned_df[feature].unique()).codes)
                        #cleaned_df[feature] = pd.Categorical(cleaned_df[feature]).codes
#---------------------------------------------------------------------------------------------------------------------------------
### Feature Transformation
#---------------------------------------------------------------------------------------------------------------------------------

                with tab4:
                
                    st.subheader("Feature Encoding",divider='blue')

                    # Function to perform feature encoding
                    @st.cache_data(ttl="2h")
                    def encode_features(data, encoder):
                        if encoder == 'Label Encoder':
                            encoder = LabelEncoder()
                            encoded_data = data.apply(encoder.fit_transform)
                        elif encoder == 'One-Hot Encoder':
                            encoder = OneHotEncoder(drop='first', sparse=False)
                            encoded_data = pd.DataFrame(encoder.fit_transform(data), columns=encoder.get_feature_names(data.columns))
                        return encoded_data
                    
                    encoding_methods = ['Label Encoder', 'One-Hot Encoder']
                    selected_encoder = st.selectbox("**Select a feature encoding method**", encoding_methods)
                    
                    encoded_df = encode_features(df, selected_encoder)
                    st.table(encoded_df.head(2))                   
                

                    st.subheader("Feature Scalling",divider='blue') 

                    # Function to perform feature scaling
                    def scale_features(data, scaler):
                        if scaler == 'Standard Scaler':
                            scaler = StandardScaler()
                        elif scaler == 'Min-Max Scaler':
                            scaler = MinMaxScaler()
                        elif scaler == 'Robust Scaler':
                            scaler = RobustScaler()

                        scaled_data = scaler.fit_transform(data)
                        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
                        return scaled_df
                    
                    scaling_methods = ['Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
                    selected_scaler = st.selectbox("**Select a feature scaling method**", scaling_methods)

                    if st.button("**Apply Feature Scalling**", key='f_scl'):
                        scaled_df = scale_features(encoded_df, selected_scaler)
                        st.table(scaled_df.head(2))
                    else:
                         df = encoded_df.copy()
#---------------------------------------------------------------------------------------------------------------------------------
### Feature Selection
#---------------------------------------------------------------------------------------------------------------------------------

                with tab5:

                    st.subheader("Feature Selection:",divider='blue')  
                    #target_variable = st.multiselect("**Target (Dependent) Variable**", df.columns)

                    col1, col2, col3 = st.columns(3) 

                    with col1:
                        #st.subheader("Feature Selection (Method 1):",divider='blue')
                        st.markdown("**Method 1 : Checking VIF Values**")
                        vif_threshold = st.number_input("**VIF Threshold**", 1.5, 10.0, 5.0)

                        @st.cache_data(ttl="2h")
                        def calculate_vif(data):
                            X = data.values
                            vif_data = pd.DataFrame()
                            vif_data["Variable"] = data.columns
                            vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                            vif_data = vif_data.sort_values(by="VIF", ascending=False)
                            return vif_data

                        # Function to drop variables with VIF exceeding the threshold
                        def drop_high_vif_variables(data, threshold):
                            vif_data = calculate_vif(data)
                            high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()
                            data = data.drop(columns=high_vif_variables)
                            return data
                                       
                        st.markdown(f"Iterative VIF Thresholding (Threshold: {vif_threshold})")
                        #X = df.drop(columns = target_variable)
                        vif_data = drop_high_vif_variables(df, vif_threshold)
                        vif_data = vif_data.drop(columns = target_variable)
                        selected_features = vif_data.columns
                        st.markdown("**Selected Features (considering VIF values in ascending orders)**")
                        st.table(selected_features)
                        #st.table(vif_data)

                    with col2:

                        #st.subheader("Feature Selection (Method 2):",divider='blue')                        
                        st.markdown("**Method 2 : Checking Selectkbest Method**")          
                        method = st.selectbox("**Select Feature Selection Method**", ["f_classif", "f_regression", "chi2", "mutual_info_classif"])
                        num_features_to_select = st.slider("**Select Number of Independent Features**", min_value=1, max_value=len(df.columns), value=5)

                        # Perform feature selection
                        if "f_classif" in method:
                            feature_selector = SelectKBest(score_func=f_classif, k=num_features_to_select)

                        elif "f_regression" in method:
                            feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)

                        elif "chi2" in method:
                            # Make sure the data is non-negative for chi2
                            df[df < 0] = 0
                            feature_selector = SelectKBest(score_func=chi2, k=num_features_to_select)

                        elif "mutual_info_classif" in method:
                            # Make sure the data is non-negative for chi2
                            df[df < 0] = 0
                            feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)

                        X = df.drop(columns = target_variable)  # Adjust 'Target' to your dependent variable
                        y = df[target_variable]  # Adjust 'Target' to your dependent variable
                        X_selected = feature_selector.fit_transform(X, y)

                        # Display selected features
                        selected_feature_indices = feature_selector.get_support(indices=True)
                        selected_features_kbest = X.columns[selected_feature_indices]
                        st.markdown("**Selected Features (considering values in 'recursive feature elimination' method)**")
                        st.table(selected_features_kbest)

                    with col3:

                        st.markdown("**Method 3 : Checking VarianceThreshold Method**") 
                        @st.cache_data(ttl="2h")   
                        def variance_threshold_feature_selection(df, threshold):
                            X = df.drop(columns=df[target_variable])  
                            selector = VarianceThreshold(threshold=threshold)
                            X_selected = selector.fit_transform(X)
                            selected_feature_indices = selector.get_support(indices=True)
                            selected_feature_names = X.columns[selected_feature_indices]
                            selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
                            return selected_df
                    
                        threshold = st.number_input("Variance Threshold", min_value=0.0, step=0.01, value=0.0)
                        selected_vth_df = variance_threshold_feature_selection(df, threshold)
                        st.markdown("**Selected Features (considering values in 'variance threshold' method)**")                    
                        st.table(selected_vth_df.head())

#---------------------------------------------------------------------------------------------------------------------------------
### Model Development & Tuning
#---------------------------------------------------------------------------------------------------------------------------------

                with tab6:

                    col1, col2, col3 = st.columns((0.3,0.3,0.6))   
                 
                    with col1:
                     
                        st.subheader("Dataset Splitting Criteria",divider='blue')
                        test_size = st.slider("**Test Size (as %)**", 10, 50, 30, 5)    
                        random_state = st.number_input("**Random State**", 0, 100, 42)
                        n_jobs = st.number_input("**Parallel Processing (n_jobs)**", -10, 10, 1)     

                    with col2: 

                        st.subheader("Algorithm",divider='blue')
                        st.info(f'Selected Algorithm : **{algorithms}**')
                        st.info(f'Target Variable : **{target_variable}**')
                        st.info(f'Target Variable Type : **{classifier_clv}**')
                      
                        #progress_text = "Prediction in progress. please wait."
                        #my_bar = st.progress(0, text=progress_text)
                        #st.button("Predict", key='Classify')
                        #with st.spinner():
                        #for percent_complete in range(100):
                            #time.sleep(0.1)
                            #my_bar.progress(percent_complete + 1, text=progress_text)

    #----------------------------------------
                        # Split the data into train and test sets
                        X = df[selected_features]
                        y = df[target_variable]
                        X_train, X_test, train_labels, test_labels = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
    #----------------------------------------

                    with col3:
                    #st.subheader("3. Tune the Hyperparameters")

                        if algorithms == 'logistic_regression':
                            st.subheader("Tune the Hyperparameters",divider='blue')
                            class_weight = st.selectbox("**Weights associated with classes in the form**", ('balanced','None'))
                            solver = st.selectbox("**Algorithm to use in the optimization problem**", ('liblinear','lbfgs', 'newton-cg', 'newton-cholesky','sag', 'saga')) 
                            max_iter = st.number_input("**Maximum number of iterations taken for the solvers to converge**", 10, 500, 100, step=10,key='max_iter')   
                            penalty = st.selectbox("**Specify the norm of the penalty**", ('l1', 'l2', 'elasticnet', 'None'))                 
                            logr = LogisticRegression(C=1.0,             
                                class_weight=class_weight, 
                                dual=False, 
                                fit_intercept=True,
                                intercept_scaling=1, 
                                l1_ratio=None, 
                                max_iter=max_iter,
                                n_jobs=n_jobs, 
                                penalty=penalty,
                                random_state=random_state, 
                                solver=solver, 
                                tol=0.0001, 
                                verbose=0,
                                warm_start=False)
                            logr.fit(X_train, train_labels)
                            logr_pred_train = logr.predict(X_train)
                            logr_pred_test = logr.predict(X_test)
                            cm_train=confusion_matrix(train_labels,logr_pred_train)
                            cm_test=confusion_matrix(test_labels,logr_pred_test)
                            models = pd.DataFrame({'Model' : ['Logistic Regression'],
                               'Training_Accuracy' : [logr.score(X_train, train_labels)],
                               'Test_Accuracy' : [logr.score(X_test, test_labels)]})
                            #actual_vs_predict = pd.DataFrame({'Actual': test_labels, 'Predicted': lin_pred})


                        if algorithms == 'decision_tree_classification':
                            st.subheader("Tune the Hyperparameters",divider='blue')
                            #criterion = st.selectbox("Measure the quality of a split", ('gini, 'entropy', 'log_loss'),key='crierian')
                            min_samples_leaf = st.number_input("**Minimum number of samples required to be at a leaf node**", 1, 10, step=1,key='min_samples_leaf')
                            min_samples_split = st.number_input("**Minimum number of samples required to split an internal node**", 2, 10, step=1,key='min_samples_split')
                            max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, step=1, key='max_depth')
                            splitter = st.radio("**Choose the split at each node**", ('best', 'random'), key='splitter')
                            dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                            min_samples_split=min_samples_split,
                                            max_depth=max_depth, 
                                            splitter = splitter)
                            dt.fit(X_train, train_labels)
                            dt_pred_train = dt.predict(X_train)
                            dt_pred_test = dt.predict(X_test)
                            cm_train=confusion_matrix(train_labels,dt_pred_train)
                            cm_test=confusion_matrix(test_labels,dt_pred_test)
                            models = pd.DataFrame({'Model' : ['Decision Tree'],
                               'Training_Score' : [dt.score(X_train, train_labels)],
                               'Test_Score' : [dt.score(X_test, test_labels)]})
        
        
                        if algorithms == 'random_forest_classification':
                            st.subheader("Tune the Hyperparameters",divider='blue')
                            min_samples_leaf = st.number_input("**Minimum number of samples required to be at a leaf node**", 1, 10, step=1,key='min_samples_leaf')
                            min_samples_split = st.number_input("**Minimum number of samples required to split an internal node**", 2, 10, step=1,key='min_samples_split')
                            n_estimators = st.number_input("**The number of trees in the forest**", 100, 5000, step=10,key='n_estimators')
                            max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, step=1, key='max_depth')
                            #bootstrap = st.radio("**Bootstrap samples when building trees**", ('True', 'False'), key='bootstrap')
                            ccp_alpha = st.number_input("**Minimal Cost-Complexity Pruning**", 0.01, 0.25, step=0.01, key='ccp_alpha')                        
                            rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf,
                                               min_samples_split=min_samples_split,
                                               n_estimators=n_estimators, 
                                               max_depth=max_depth, 
                                               bootstrap=True,
                                               ccp_alpha=ccp_alpha,
                                               n_jobs=n_jobs)
                            rf.fit(X_train, train_labels)
                            rf_pred_train = rf.predict(X_train)
                            rf_pred_test = rf.predict(X_test)
                            cm_train=confusion_matrix(train_labels,rf_pred_train)
                            cm_test=confusion_matrix(test_labels,rf_pred_test)
                            models = pd.DataFrame({'Model' : ['Random Forest'],
                               'Training_Score' : [rf.score(X_train, train_labels)],
                               'Test_Score' : [rf.score(X_test, test_labels)]})
 
        
                        if algorithms == 'gradient_boosting':
                            st.subheader("Tune the Hyperparameters",divider='blue')
                            min_samples_leaf = st.number_input("**Minimum number of samples required to be at a leaf node**", 1, 50, 10,step=5,key='min_samples_leaf')
                            min_samples_split = st.number_input("**Minimum number of samples required to split an internal node**", 1, 50, 10,step=5,key='min_samples_split')
                            n_estimators = st.number_input("**The number of trees in the forest**", 100, 500, 100,step = 10,key='n_estimators')
                            max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, 10,step = 1, key='max_depth')
                            learning_rate = st.number_input("**Learning rate**", .01, .1, 0.1,step =.01, key ='learning_rate')
                            subsample = st.number_input("**The fraction of samples fitting to indivitual base learners**", 0.1, 1.0, 1.0,step =.01, key ='subsample')
                            gb = GradientBoostingClassifier(min_samples_leaf=min_samples_leaf,
                                               min_samples_split=min_samples_split,
                                                n_estimators=n_estimators, 
                                                max_depth=max_depth, 
                                                learning_rate=learning_rate,
                                                subsample=subsample)
                            gb.fit(X_train, train_labels)
                            gb_pred_train = gb.predict(X_train)
                            gb_pred_test = gb.predict(X_test)
                            cm_train=confusion_matrix(train_labels,gb_pred_train)
                            cm_test=confusion_matrix(test_labels,gb_pred_test)
                            models = pd.DataFrame({'Model' : ['Gradient Boosting'],
                                                    'Training_Score' : [gb.score(X_train, train_labels)],
                                                    'Test_Score' : [gb.score(X_test, test_labels)]})
         
        
                        if algorithms == 'xtreme_gradient_boosting':
                            st.subheader("Tune the Hyperparameters",divider='blue')
                            n_estimators = st.number_input("**The number of trees in the forest**", 100, 500, 100,step = 10,key='n_estimators')
                            max_depth = st.number_input("**The maximum depth of the tree**", 1, 20, 10,step = 1, key='max_depth')
                            learning_rate = st.number_input("**Learning rate**", .01, .1, 0.1,step =.01, key ='learning_rate')
                            booster= st.radio("**Boosting options**", ('gbtree', 'gblinear'), key='booster')
                            xgb = XGBClassifier(n_estimators=n_estimators, 
                                    max_depth=max_depth, 
                                    learning_rate=learning_rate, 
                                    booster = booster,
                                    colsample_bylevel = 1, 
                                    colsample_bytree = 1,
                                    gamma = 0,  
                                    min_child_weight = 1, 
                                    objective ='binary:logistic', 
                                    reg_alpha = 1, 
                                    reg_lambda = 1,
                                    scale_pos_weight=1,  
                                    subsample = 1)
                            xgb.fit(X_train, train_labels)
                            xgb_pred_train = xgb.predict(X_train)
                            gb_pred_test = xgb.predict(X_test)
                            cm_train=confusion_matrix(train_labels,xgb_pred_train)
                            cm_test=confusion_matrix(test_labels,xgb_pred_test)      
                            models = pd.DataFrame({'Model' : ['Xtreme Gradient Boosting'],
                                                    'Training_Score' : [xgb.score(X_train, train_labels)],
                                                    'Test_Score' : [xgb.score(X_test, test_labels)]})

#---------------------------------------------------------------------------------------------------------------------------------
### Model Performance
#---------------------------------------------------------------------------------------------------------------------------------
               
                with tab7:

                    col1, col2, col3, col4 = st.columns((0.3,0.3,0.5,0.5)) 

                    with col1:
                        with st.container():
                            st.subheader("Model Accuracy",divider='blue')
                            st.table(models.head())
        
                            if st.checkbox("**Show Confusion matrix**"):
                                st.write('**Confusion matrix (Train Dataset):**', cm_train)
                                st.write('**Confusion matrix (Test Dataset):**', cm_test)

                    with col2:
                        with st.container():
                            if algorithms == 'logistic_regression':
                                st.subheader("Model Report",divider='blue')
                                logr_pred_train = logr.predict(X_train)
                                logr_pred_test = logr.predict(X_test)
                
                                st.markdown("**Report - Training Dataset**")
                                train_report = classification_report(train_labels, logr_pred_train,output_dict = True)
                                st.write(pd.DataFrame(train_report).transpose().head(6))

                                st.markdown("**Report - Test Dataset**")
                                test_report = classification_report(test_labels, logr_pred_test,output_dict = True)
                                st.write(pd.DataFrame(test_report).transpose().head(6))
                
                
                            elif algorithms == 'decision_tree_classification':
                                st.subheader("Model Report",divider='blue')
                                dt_pred_train = dt.predict(X_train)
                                dt_pred_test = dt.predict(X_test)
                
                                st.markdown("**Report - Training Dataset**")
                                train_report = classification_report(train_labels, dt_pred_train,output_dict = True)
                                st.write(pd.DataFrame(train_report).transpose().head(6),use_container_width = False)

                                st.markdown("**Report - Test Dataset**")
                                test_report = classification_report(test_labels, dt_pred_test,output_dict = True)
                                st.write(pd.DataFrame(test_report).transpose().head(6),use_container_width = False)

            
                            elif algorithms == 'random_forest_classification':
                                st.subheader("Model Report",divider='blue')
                                rf_pred_train = rf.predict(X_train)
                                rf_pred_test = rf.predict(X_test)
                
                                st.markdown("**Report - Training Dataset**")
                                train_report = classification_report(train_labels, rf_pred_train,output_dict = True)
                                st.write(pd.DataFrame(train_report).transpose().head(6),use_container_width = False)

                                st.markdown("**Report - Test Dataset**")
                                test_report = classification_report(test_labels, rf_pred_test,output_dict = True)
                                st.write(pd.DataFrame(test_report).transpose().head(6),use_container_width = False)

            
                            elif algorithms == 'gradient_boosting':
                                st.subheader("Model Report",divider='blue')
                                gb_pred_train = gb.predict(X_train)
                                gb_pred_test = gb.predict(X_test)
                
                                st.markdown("**Report - Training Dataset**")
                                train_report = classification_report(train_labels, gb_pred_train,output_dict = True)
                                st.write(pd.DataFrame(train_report).transpose().head(6),use_container_width = False)

                                st.markdown("**Report - Test Dataset**")
                                test_report = classification_report(test_labels, gb_pred_test,output_dict = True)
                                st.write(pd.DataFrame(test_report).transpose().head(6),use_container_width = False)

            
                            elif algorithms == 'xtreme_gradient_boosting':
                                st.subheader("5. Model Report",divider='blue')
                                xgb_pred_train = xgb.predict(X_train)
                                xgb_pred_test = xgb.predict(X_test)
                
                                st.markdown("**Report - Training Dataset**")
                                train_report = classification_report(train_labels, xgb_pred_train,output_dict = True)
                                st.write(pd.DataFrame(train_report).transpose().head(6),use_container_width = False)

                                st.markdown("**Report - Test Dataset**")
                                test_report = classification_report(test_labels, xgb_pred_test,output_dict = True)
                                st.write(pd.DataFrame(test_report).transpose().head(6),use_container_width = False)
 

                    with col3:
                        with st.container():

                                    if algorithms == 'logistic_regression':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_train = logr.predict_proba(X_train)[:, 1]                    
                                        log_train_auc = roc_auc_score(train_labels, probs_train)
                                        fpr, tpr, thresholds = roc_curve(train_labels, probs_train)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(log_train_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Train Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Train): %.3f**' % log_train_auc)

                                    if algorithms == 'decision_tree_classification':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_train = dt.predict_proba(X_train)[:, 1]                    
                                        dt_train_auc = roc_auc_score(train_labels, probs_train)
                                        fpr, tpr, thresholds = roc_curve(train_labels, probs_train)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(dt_train_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Train Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Train): %.3f**' % dt_train_auc)

                                    if algorithms == 'random_forest_classification':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_train = rf.predict_proba(X_train)[:, 1]                    
                                        rf_train_auc = roc_auc_score(train_labels, probs_train)
                                        fpr, tpr, thresholds = roc_curve(train_labels, probs_train)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(rf_train_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Train Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Train): %.3f**' % rf_train_auc)

                                    if algorithms == 'gradient_boosting':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_train = gb.predict_proba(X_train)[:, 1]                    
                                        gb_train_auc = roc_auc_score(train_labels, probs_train)
                                        fpr, tpr, thresholds = roc_curve(train_labels, probs_train)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(gb_train_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Train Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Train): %.3f**' % gb_train_auc)

                                    if algorithms == 'xtreme_gradient_boosting':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_train = xgb.predict_proba(X_train)[:, 1]                    
                                        xgb_train_auc = roc_auc_score(train_labels, probs_train)
                                        fpr, tpr, thresholds = roc_curve(train_labels, probs_train)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(xgb_train_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Train Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Train): %.3f**' % xgb_train_auc)      

                    with col4:
                        with st.container():

                                    if algorithms == 'logistic_regression':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_test = logr.predict_proba(X_test)[:, 1] 
                                        log_test_auc = roc_auc_score(test_labels, probs_test)
                                        fpr, tpr, thresholds = roc_curve(test_labels, probs_test)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(log_test_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Test): %.3f**' % log_test_auc)

                                    if algorithms == 'decision_tree_classification':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_test = dt.predict_proba(X_test)[:, 1] 
                                        dt_test_auc = roc_auc_score(test_labels, probs_test)
                                        fpr, tpr, thresholds = roc_curve(test_labels, probs_test)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(dt_test_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Test): %.3f**' % dt_test_auc)

                                    if algorithms == 'random_forest_classification':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_test = rf.predict_proba(X_test)[:, 1] 
                                        rf_test_auc = roc_auc_score(test_labels, probs_test)
                                        fpr, tpr, thresholds = roc_curve(test_labels, probs_test)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(rf_test_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Test): %.3f**' % rf_test_auc)

                                    if algorithms == 'gradient_boosting':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_test = gb.predict_proba(X_test)[:, 1] 
                                        gb_test_auc = roc_auc_score(test_labels, probs_test)
                                        fpr, tpr, thresholds = roc_curve(test_labels, probs_test)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(gb_test_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Test): %.3f**' % gb_test_auc)

                                    if algorithms == 'xtreme_gradient_boosting':
                                        st.subheader("ROC - AUC Graph",divider='blue') 
                                        probs_test = xgb.predict_proba(X_test)[:, 1] 
                                        xgb_test_auc = roc_auc_score(test_labels, probs_test)
                                        fpr, tpr, thresholds = roc_curve(test_labels, probs_test)
                                        fig, ax = plt.subplots()
                                        ax.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(xgb_test_auc))
                                        ax.plot([0, 1], [0, 1], 'k--')
                                        ax.set_xlabel('False Positive Rate')
                                        ax.set_ylabel('True Positive Rate')
                                        ax.set_title('ROC Curve (Test Dataset)')
                                        ax.legend(loc="lower right")
                                        st.pyplot(fig,use_container_width = True)
                                        st.write('**AUC | (Test): %.3f**' % xgb_test_auc)         

                     
                    st.divider()
                    st.subheader("Feature Importance",divider='blue')
    
                    col1, col2, col3 = st.columns(3) 

                    with col1:
                        with st.container():
                            if algorithms == 'logistic_regression':
                                st.markdown("**Method 1 : Impurity-based Importance**")
                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="ib_lr")
                                coefficients = pd.DataFrame({'Feature': selected_features, 'Coefficient': logr.coef_[0]})
                                coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
                                st.bar_chart(coefficients.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(coefficients.head(top_features_count))                            

                            if algorithms == 'decision_tree_classification':
                                st.markdown("**Method 1 : Impurity-based Importance**")                    
                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="ib_dt")
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': dt.feature_importances_})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                            if algorithms == 'random_forest_classification':
                                st.markdown("**Method 1 : Impurity-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="ib_dt")
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': rf.feature_importances_})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                            if algorithms == 'gradient_boosting':
                                st.markdown("**Method 1 : Impurity-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="ib_dt")
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': gb.feature_importances_})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                            if algorithms == 'xtreme_gradient_boosting':
                                st.markdown("**Method 1 : Impurity-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="ib_dt")
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': xgb.feature_importances_})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                    with col2:
                        with st.container():

                            if algorithms == 'logistic_regression':
                                st.markdown("**Method 2 : Permutation-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="pb_lr")
                                result = permutation_importance(logr, X_test, test_labels, n_repeats=30, random_state=random_state)
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': result.importances_mean})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                            if algorithms == 'decision_tree_classification':
                                st.markdown("**Method 2 : Permutation-based Importance**")   

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="pb_dt")
                                result = permutation_importance(dt, X_test, test_labels, n_repeats=30, random_state=random_state)
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': result.importances_mean})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                                st.write("Change in Accuracy for Each Feature")
                                original_accuracy = accuracy_score(test_labels, dt.predict(X_test))
                                accuracy_changes = result.importances_mean - original_accuracy
                                st.bar_chart(pd.DataFrame({'Feature': selected_features, 'Accuracy Change': accuracy_changes}).set_index('Feature'))

                            if algorithms == 'random_forest_classification':
                                st.markdown("**Method 2 : Permutation-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="pb_dt")
                                result = permutation_importance(rf, X_test, test_labels, n_repeats=30, random_state=random_state)
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': result.importances_mean})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                                st.write("Change in Accuracy for Each Feature")
                                original_accuracy = accuracy_score(test_labels, rf.predict(X_test))
                                accuracy_changes = result.importances_mean - original_accuracy
                                st.bar_chart(pd.DataFrame({'Feature': selected_features , 'Accuracy Change': accuracy_changes}).set_index('Feature'))

                            if algorithms == 'gradient_boosting':
                                st.markdown("**Method 2 : Permutation-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="pb_dt")
                                result = permutation_importance(gb, X_test, test_labels, n_repeats=30, random_state=random_state)
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': result.importances_mean})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                                st.write("Change in Accuracy for Each Feature")
                                original_accuracy = accuracy_score(test_labels, gb.predict(X_test))
                                accuracy_changes = result.importances_mean - original_accuracy
                                st.bar_chart(pd.DataFrame({'Feature': selected_features, 'Accuracy Change': accuracy_changes}).set_index('Feature'))

                            if algorithms == 'xtreme_gradient_boosting':
                                st.markdown("**Method 2 : Permutation-based Importance**")

                                top_features_count = st.slider("**Top N Features to Display**", 1, len(selected_features), 10, key="pb_dt")
                                result = permutation_importance(xgb, X_test, test_labels, n_repeats=30, random_state=random_state)
                                feature_importance = pd.DataFrame({'Feature': selected_features, 'Importance': result.importances_mean})
                                feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
                                st.bar_chart(feature_importance.set_index('Feature'))

                                st.write("**Top Features:**")
                                st.write(feature_importance.head(top_features_count))

                                st.write("Change in Accuracy for Each Feature")
                                original_accuracy = accuracy_score(test_labels, xgb.predict(X_test))
                                accuracy_changes = result.importances_mean - original_accuracy
                                st.bar_chart(pd.DataFrame({'Feature': selected_features, 'Accuracy Change': accuracy_changes}).set_index('Feature'))

                    with col3:
                        with st.container():
                            if algorithms == 'logistic_regression':
                                st.markdown("**Method 3 : SHAP-based Importance**")

                                #explainer = shap.TreeExplainer(logr)
                                explainer = shap.LinearExplainer(logr, X_train, feature_dependence="independent")
                                shap_values = explainer.shap_values(X_test)
                                #instance_index = st.slider("**Select Instance Index**", 0, len(X_test) - 1)
                                #X_test_array = X_test.toarray()

                                # Display summary plot for feature importance
                                st.write("**SHAP Summary Plot for Feature Importance**")
                                fig, ax = plt.subplots()
                                #ax = shap.summary_plot(shap_values[1], X_test, feature_names=selected_features)
                                ax = shap.summary_plot(shap_values, X_test, feature_names=selected_features)
                                st.pyplot(fig,use_container_width = True)

                            if algorithms == 'decision_tree_classification':
                                st.markdown("**Method 3 : SHAP-based Importance**")

                                explainer = shap.TreeExplainer(dt)
                                shap_values = explainer.shap_values(X_test)
                                instance_index = st.slider("**Select Instance Index**", 0, len(X_test) - 1)

                                # Display summary plot for feature importance
                                st.write("**SHAP Summary Plot for Feature Importance**")
                                fig, ax = plt.subplots()
                                ax = shap.summary_plot(shap_values[1], X_test, feature_names=selected_features)
                                st.pyplot(fig,use_container_width = True)

                            if algorithms == 'random_forest_classification':
                                st.markdown("**Method 3 : SHAP-based Importance**")

                                explainer = shap.TreeExplainer(rf)
                                shap_values = explainer.shap_values(X_test)
                                instance_index = st.slider("**Select Instance Index**", 0, len(X_test) - 1)

                                # Display summary plot for feature importance
                                st.write("**SHAP Summary Plot for Feature Importance**")
                                fig, ax = plt.subplots()
                                ax = shap.summary_plot(shap_values, X_test, plot_type="bar")
                                st.pyplot(fig,use_container_width = True)

                            if algorithms == 'gradient_boosting':
                                st.markdown("**Method 3 : SHAP-based Importance**")

                                explainer = shap.TreeExplainer(gb)
                                shap_values = explainer.shap_values(X_test)
                                instance_index = st.slider("**Select Instance Index**", 0, len(X_test) - 1)

                                # Display summary plot for feature importance
                                st.write("**SHAP Summary Plot for Feature Importance**")
                                fig, ax = plt.subplots()
                                ax = shap.summary_plot(shap_values, X_test, plot_type="bar")
                                st.pyplot(fig,use_container_width = True)

                            if algorithms == 'xtreme_gradient_boosting':
                                st.markdown("**Method 3 : SHAP-based Importance**")

                                explainer = shap.TreeExplainer(xgb)
                                shap_values = explainer.shap_values(X_test)
                                instance_index = st.slider("**Select Instance Index**", 0, len(X_test) - 1)

                                # Display summary plot for feature importance
                                st.write("**SHAP Summary Plot for Feature Importance**")
                                fig, ax = plt.subplots()
                                ax = shap.summary_plot(shap_values, X_test, plot_type="bar")
                                st.pyplot(fig,use_container_width = True)
#---------------------------------------------------------------------------------------------------------------------------------
### Model Validation
#---------------------------------------------------------------------------------------------------------------------------------

                with tab8:

                    st.subheader("Cross Validation", divider='blue')
                    col1, col2 = st.columns(2) 
                    with col1:

                        cv = st.slider("**CV Value**", 0, 10, 5, 1)  
                        scoring = st.selectbox("**Select type of scoring**",["accuracy"])
                    
                        st.divider()

                        if algorithms == 'logistic_regression':
                            cv_score = cross_val_score(logr, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation Accuracy**: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

                        if algorithms == 'decision_tree_classification':
                            cv_score = cross_val_score(dt, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation Accuracy**: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

                        if algorithms == 'random_forest_classification':
                            cv_score = cross_val_score(rf, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation Accuracy**: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

                        if algorithms == 'gradient_boosting':
                            cv_score = cross_val_score(gb, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                            st.write(f"**Cross-Validation Accuracy**: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

                    if algorithms == 'xtreme_gradient_boosting':
                        cv_score = cross_val_score(xgb, 
                                       X_train, train_labels, 
                                       cv=cv, 
                                       scoring=scoring)
                        st.write(f"**Cross-Validation Accuracy**: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

                    with col2:  
                         
                        st.write("### Cross-Validation Plot:")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax = plt.plot(range(1, cv + 1), cv_score, marker='o')
                        #plt.title('Cross-Validation Mean Squared Error Scores')
                        plt.xlabel('Fold')
                        plt.ylabel('accuracy')
                        st.pyplot(fig)

                    st.subheader("Bias-Variance Tradeoff", divider='blue')
                    col1, col2 = st.columns(2) 
                    with col1:
                        degree = st.slider("**Polynomial Degree**", min_value=1, max_value=10, value=3)

                    with col2:
                        noise_level = st.slider("**Noise Level**", min_value=0.1, max_value=5.0, value=1.0)
#---------------------------------------------------------------------------------------------------------------------------------
### Model Cross Check
#---------------------------------------------------------------------------------------------------------------------------------
