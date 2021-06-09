#!/usr/bin/env python
# coding: utf-8

# # Prices of Airline Flight Tickets

# # Import Relevant Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Load the Data

# In[2]:


train_data=pd.read_excel('Data_Train.xlsx')


# In[3]:


train_data.head()


# # Data Preprocessing

# In[4]:


train_data.isna().sum()


# In[5]:


train_data.shape


# In[6]:


train_data.dropna(inplace=True)


# In[7]:


train_data.isna().sum()


# In[8]:


train_data.dtypes


# In[9]:


def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])
    


# In[10]:


train_data.columns


# In[11]:


for i in ['Date_of_Journey','Dep_Time','Arrival_Time',]:
    change_into_datetime(i)


# In[12]:


train_data.dtypes


# In[13]:


train_data['journey_day']=train_data['Date_of_Journey'].dt.day
train_data['journey_month']=train_data['Date_of_Journey'].dt.month


# In[14]:


train_data.head()


# In[15]:


train_data.drop('Date_of_Journey',axis=1,inplace=True)


# In[16]:


def extract_hour(df,col):
    df[col+"_hour"]=df[col].dt.hour


# In[17]:


def extract_min(df,col):
    df[col+"_minute"]=df[col].dt.minute


# In[18]:


def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)


# In[19]:


extract_hour(train_data,'Dep_Time')


# In[20]:


extract_min(train_data,'Dep_Time')


# In[21]:


drop_column(train_data,'Dep_Time')


# In[22]:


train_data.head()


# In[23]:


extract_hour(train_data,'Arrival_Time')

extract_min(train_data,'Arrival_Time')

drop_column(train_data,'Arrival_Time')


# In[24]:


train_data.head()


# In[25]:


'2h 50m'.split(' ')


# In[26]:


duration=list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:               
            duration[i]=duration[i] + ' 0m'     
        else:
            duration[i]='0h '+ duration[i]    
    


# In[27]:


train_data['Duration']=duration


# In[28]:


train_data.head()


# In[29]:


'2h 50m'.split(' ')[1][0:-1]


# In[30]:


def hour(x):
    return x.split(' ')[0][0:-1]


# In[31]:


def min(x):
    return x.split(' ')[1][0:-1]


# In[32]:


train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(min)


# In[33]:


train_data.head()


# In[34]:


train_data.drop('Duration',axis=1,inplace=True)


# In[35]:


train_data.head()


# In[36]:


train_data.dtypes


# In[37]:


train_data['Duration_hours']=train_data['Duration_hours'].astype(int)
train_data['Duration_mins']=train_data['Duration_mins'].astype(int)


# In[38]:


train_data.dtypes


# In[39]:


train_data.head()


# In[40]:


cat_col=[col for col in train_data.columns if train_data[col].dtype=='O']
cat_col


# In[41]:


cont_col=[col for col in train_data.columns if train_data[col].dtype!='O']
cont_col


# # Handling Categorical Data

# In[42]:


categorical=train_data[cat_col]
categorical.head()


# In[43]:


categorical['Airline'].value_counts()


# ### Airline vs Price Analysis

# In[44]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Airline',data=train_data.sort_values('Price',ascending=False))


# In[45]:


train_data.head()


# ### Perform Total_Stops vs Price Analysis

# In[46]:


plt.figure(figsize=(15,5))
sns.boxplot(y='Price',x='Total_Stops',data=train_data.sort_values('Price',ascending=False))


# In[47]:


len(categorical['Airline'].unique())


# In[48]:


Airline=pd.get_dummies(categorical['Airline'], drop_first=True)
Airline.head()


# In[49]:


categorical['Source'].value_counts()


# In[50]:


plt.figure(figsize=(15,5))
sns.catplot(y='Price',x='Source',data=train_data.sort_values('Price',ascending=False),kind='boxen')


# In[51]:


Source=pd.get_dummies(categorical['Source'], drop_first=True)
Source.head()


# In[52]:


categorical['Destination'].value_counts()


# In[53]:


Destination=pd.get_dummies(categorical['Destination'], drop_first=True)
Destination.head()


# In[54]:


import warnings
warnings.filterwarnings('ignore')


# In[55]:


categorical['Route']


# In[56]:


categorical['Route_1']=categorical['Route'].str.split('→').str[0]
categorical['Route_2']=categorical['Route'].str.split('→').str[1]
categorical['Route_3']=categorical['Route'].str.split('→').str[2]
categorical['Route_4']=categorical['Route'].str.split('→').str[3]
categorical['Route_5']=categorical['Route'].str.split('→').str[4]


# In[57]:


categorical.head()


# In[58]:


import warnings 
from warnings import filterwarnings
filterwarnings('ignore')


# In[59]:


categorical['Route_1'].fillna('None',inplace=True)
categorical['Route_2'].fillna('None',inplace=True)
categorical['Route_3'].fillna('None',inplace=True)
categorical['Route_4'].fillna('None',inplace=True)
categorical['Route_5'].fillna('None',inplace=True)


# In[60]:


categorical.head()


# In[61]:


for feature in categorical.columns:
    print('{} has total {} categories \n'.format(feature,len(categorical[feature].value_counts())))


# In[62]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[63]:


categorical.columns


# In[64]:


for i in ['Route_1', 'Route_2', 'Route_3', 'Route_4','Route_5']:
    categorical[i]=encoder.fit_transform(categorical[i])


# In[65]:


categorical.head()


# In[66]:


drop_column(categorical,'Route')
drop_column(categorical,'Additional_Info')


# In[67]:


categorical.head()


# In[68]:


categorical['Total_Stops'].value_counts()


# In[69]:


categorical['Total_Stops'].unique()


# In[70]:


dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}


# In[71]:


categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[72]:


categorical.head()


# In[73]:


train_data[cont_col]


# In[74]:


data_train=pd.concat([categorical,Airline,Source,Destination,train_data[cont_col]],axis=1)
data_train.head()


# In[75]:


drop_column(data_train,'Airline')
drop_column(data_train,'Source')
drop_column(data_train,'Destination')


# In[76]:


data_train.head()


# In[77]:


pd.set_option('display.max_columns',35)


# In[78]:


data_train.head()


# In[79]:


data_train.columns


# # Outlier detection

# In[80]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    


# In[81]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# ### Dealing with Outliers

# In[82]:


data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])


# In[83]:


plt.figure(figsize=(30,20))
plot(data_train,'Price')


# In[84]:


X=data_train.drop('Price',axis=1)
X.head()


# In[85]:


y=data_train['Price']
y


# In[86]:


type(X)


# In[87]:


type(y)


# In[88]:


X.isnull().sum()


# In[89]:


y.isnull().sum()


# In[90]:


X.isnull().sum()


# In[91]:


X.shape


# In[92]:


y.shape


# # Feature Selection

# In[93]:


np.array(X)


# In[94]:


np.array(y)


# In[95]:


from sklearn.feature_selection import mutual_info_classif


# In[96]:


mutual_info_classif(X,y)


# In[97]:


X.dtypes


# In[98]:


imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
imp


# In[99]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# ## Split dataset into train & test

# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# # Model Building using ML

# In[102]:


from sklearn import metrics

import pickle
def predict(ml_model,dump):
    model=ml_model.fit(X_train,y_train)
    print('Training score : {}'.format(model.score(X_train,y_train)))
    y_prediction=model.predict(X_test)
    print('predictions are: \n {}'.format(y_prediction))
    print('\n')
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score: {}'.format(r2_score))
    print('MAE:',metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE:',metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    sns.distplot(y_test-y_prediction)
    
    if dump==1:
        file=open('model.pkl','wb')
        pickle.dump(model,file)


# In[103]:


from sklearn.ensemble import RandomForestRegressor


# In[104]:


predict(RandomForestRegressor(),1)


# # Testing multiple Algorithms

# In[105]:



from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[106]:


predict(DecisionTreeRegressor(),0)


# In[107]:


predict( KNeighborsRegressor(),0)


# In[108]:


predict(LinearRegression(),0)


# In[109]:


reg_rf=RandomForestRegressor()


# # Hyperparameter Tuning

# In[110]:


from sklearn.model_selection import RandomizedSearchCV


# In[111]:


n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=6)]


max_features=['auto','sqrt']


max_depth=[int(x) for x in np.linspace(start=5,stop=30,num=4)]

min_samples_split=[5,10,15,100]


# In[112]:


random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split
}


# In[113]:


random_grid


# In[114]:


rf_random=RandomizedSearchCV(estimator=reg_rf,param_distributions=random_grid,cv=3,verbose=2,n_jobs=-1)


# In[115]:


rf_random.fit(X_train,y_train)


# In[116]:


rf_random.best_params_


# In[117]:


prediction=rf_random.predict(X_test)


# In[118]:


sns.distplot(y_test-prediction)


# In[119]:


metrics.r2_score(y_test,prediction)


# In[120]:


print('MAE',metrics.mean_absolute_error(y_test,prediction))
print('MSE',metrics.mean_squared_error(y_test,prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# # Save the model to reuse it again

# In[121]:


import pickle


# In[122]:


file=open('rf_random.pkl','wb')
pickle.dump(rf_random,file)


# ### Comparing the new model with previous one

# In[123]:


Pmodel=open('model.pkl','rb')
Pforest=pickle.load(Pmodel)


# In[124]:


y_prediction=Pforest.predict(X_test)


# In[125]:


y_prediction


# In[126]:


metrics.r2_score(y_test,y_prediction)


# In[127]:


Nmodel=open('rf_random.pkl','rb')
Nforest=pickle.load(Nmodel)


# In[128]:


y_prediction=Nforest.predict(X_test)


# In[129]:


y_prediction


# In[130]:


metrics.r2_score(y_test,y_prediction)

