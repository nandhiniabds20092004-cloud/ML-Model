import numpy as np
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from analysis import generate_summary, suggest_improvements

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingClassifier, GradientBoostingRegressor)

from sklearn.metrics import (mean_squared_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)

model = genai.GenerativeModel('gemini-2.5-flash-lite')

st.set_page_config("ML Moldel", page_icon='👩🏻‍💻', layout='wide')
st.title(":green[ML Model Automation]🤖🧠🇦🇮👾")
st.header('Streamlit App to get CSV and target as input and performs ML algoriths')


uploaded_file=st.file_uploader('Upload your file here 📝',type=['csv'])

if  uploaded_file:
    st.markdown('### Preview 📋')
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox(':blue[Select target 🎯]', df.columns)

    st.write(f" :red[Taget variable :] {target}")

    if target:

        x = df.drop(columns = [target]).copy()
        y = df[target].copy()

        # PREPROCESSING
        # ==========================

        num_cols = x.select_dtypes(include= np.number).columns.to_list()
        cat_cols = x.select_dtypes(include= ['object']).columns.to_list()

        # MISSiNG VALUE TREATMENT

        x[num_cols] = x[num_cols].fillna(x[num_cols].median())
        x[cat_cols] = x[cat_cols].fillna('Missing data')

        # Encoding

        x = pd.get_dummies(data= x, columns = cat_cols, drop_first= True, dtype= int)

        # for categoric target

        
        if y.dtype == 'object':
            label = LabelEncoder()
            y = label.fit_transform(y)

        # DETECT THE PROBLEM TYPE

        if df[target].dtype == 'object' or len(np.unique(y)) <= 10:
            problem_type = 'Classification'

        else:
            problem_type = 'Reggression'

        st.write(f"### 🔍 PROBLEM TYPE : {problem_type}" )

        # TRAIN TEST SPLIT

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,
                                                        random_state=42)
         
        # SCALING
        # fit_transform on train data
        # transform in test data
        
        for i in xtrain.columns:
            s = StandardScaler()
             
            xtrain[i] = s.fit_transform(xtrain[[i]])
            xtest[i] = s.transform(xtest[[i]])
             
        # MODELS
        # =====
        
        results = []
         
        if problem_type == 'Regression':
            models = {'Linear Regression': LinearRegression(),
                      'Random Forest': RandomForestRegressor(random_state=42),
                      'Gradient Boosting': GradientBoostingRegressor(random_state=42)}
             
            for name, model in models.items():
                model.fit(xtrain, ytrain)
                ypred = model.predict(xtest)
                 
                results.append({'Model Name': name, 
                                'R2 Score': round(r2_score(ytest, ypred),3),
                                'MSE': round(mean_squared_error(ytest,ypred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(ytest,ypred)),3)})
        else:
              
            models = {'Logistic Regression': LogisticRegression(), 
                     'Random Forest': RandomForestClassifier(random_state= 42),
                     'Gradient Boosting': GradientBoostingClassifier(random_state= 42)}
            for name, model in models.items():
                model.fit(xtrain,ytrain)
                ypred = model.predict(xtest)
                
                results.append({'Model Name': name,
                                'Accuracy': round(accuracy_score(ytest, ypred), 3),
                                'Precision': round(precision_score(ytest, ypred, average='weighted'), 3),
                                'Recall': round(recall_score(ytest, ypred, average='weighted'), 3),
                                'F1 Score': round(f1_score(ytest, ypred, average='weighted'), 3)
                                })
                
        results_df = pd.DataFrame(results)
        st.write(f'### :green[Results] 📊')
        st.dataframe(results_df)               


    if problem_type == 'Regression' :
        st.bar_chart(results_df.set_index('Model Name')['R2 Score'])
        st.bar_chart(results_df.set_index('Model Name')['RMSE'])
    else:
        st.bar_chart(results_df.set_index('Model Name')['Accuracy'])
        st.bar_chart(results_df.set_index('Model Name')['F1 Score'])

    if st.button('Generate Summary'):
        summary = generate_summary(results_df)
        st.write(summary)

    if st.button('Suggest Improvements'):
        suggest = suggest_improvements(results_df)
        st.write(suggest)