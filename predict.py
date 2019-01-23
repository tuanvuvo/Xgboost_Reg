import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer
import requests
from sklearn.externals import joblib

model_columns = joblib.load("model_columns_xgboost.pkl") 
print ('Model columns loaded')
import pickle
loaded_model = pickle.load(open("xgb-profit-quote.dat", "rb"))
Select_Columns = ['Cargo Source', 'From','To','Bound','Mode','Commodity','Vol','Term']

df_test = pd.read_csv('test.csv')
selected_df = df_test[Select_Columns]
selected_df = pd.get_dummies(data=selected_df,columns=['Term','From','To','Bound','Mode','Commodity','Cargo Source']) #,'GNT Staff','Customer Name'
selected_df = selected_df.reindex(columns=model_columns, fill_value=0)
X = np.array(selected_df,dtype="float64") #
scaler = StandardScaler()
X= scaler.fit_transform(X)
#print(X.shape)
#print(X)
y_pred = loaded_model.predict(X)
print(y_pred)


