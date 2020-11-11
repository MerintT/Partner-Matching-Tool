import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datetime import datetime
import configparser

config = configparser.ConfigParser()
config.read("configm.ini")

df_Main = pd.read_csv('https://app.redash.io/xchange/api/queries/464708/results.csv?api_key=' + config.get('data-extraction','trainData'))
df_Main = df_Main.fillna(0)


df_drop = pd.read_csv('https://app.redash.io/xchange/api/queries/381074/results.csv?api_key=' + config.get('data-extraction','dropoffKey'))
df_drop = df_drop.rename(columns={'Dropoff': 'Dropoff_location'})

df_pick = pd.read_csv('https://app.redash.io/xchange/api/queries/381097/results.csv?api_key=' + config.get('data-extraction','pickupKey'))
df_pick = df_pick.rename(columns={'Pickup': 'Pickup_location'})

df_requester = pd.read_csv('https://app.redash.io/xchange/api/queries/381058/results.csv?api_key=' + config.get('data-extraction','requesterKey'))
df_requester = df_requester.rename(columns={'requester_id': 'Requester'})

df_add = pd.read_csv('https://app.redash.io/xchange/api/queries/381054/results.csv?api_key=' + config.get('data-extraction','addresseeKey'))
df_add = df_add.rename(columns={'Addressee_id': 'Addressee'})

def dataMerge(df1,df2,col):
	result = pd.merge(df1, df2, how='left', on=[col])
	return result

result1 = dataMerge(df1=df_Main,df2=df_drop,col='Dropoff_location')
result1.drop(["Dropoff_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result1 = result1.rename(columns={'Shares_accepted': 'Dropoff_location'}) # Renaming columns

result2 = dataMerge(df1=result1,df2=df_pick,col='Pickup_location')
result2.drop(["Pickup_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result2 = result2.rename(columns={'Shares_accepted': 'Pickup_location'}) # Renaming columns

result3 = dataMerge(df1=result2,df2=df_requester,col='Requester')
result3.drop(["Requester","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result3 = result3.rename(columns={'Frequency': 'Requester'})# Renaming columns

result4 = dataMerge(df1=result3,df2=df_add,col='Addressee')
result4.drop(["Addressee","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result4 = result4.rename(columns={'Frequency': 'Addressee'})# Renaming columns

result4 = result4.fillna(0)

result7= result4

result7.drop(['request_id'], axis=1 , inplace=True)

result7 = pd.concat([result7,pd.get_dummies(result7['Direction'])],axis=1)
result7.drop(['Direction'],axis=1, inplace=True)

#result7 = pd.concat([result7,pd.get_dummies(result7['Container_Type'])],axis=1)
#result7.drop(['Container_Type'],axis=1, inplace=True)

dep = result7.Deal_status
dep_df = pd.DataFrame(dep)

result7.drop(['Deal_status'],axis=1, inplace=True)

result7 = pd.concat([result7,dep_df],axis=1)

result7_majority = result7[result7.Deal_status==0]
result7_minority = result7[result7.Deal_status==1]

samples = result7_minority.shape[0]

result7_majority_downsampled = resample(result7_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=samples,     # to match minority class
                                 random_state=123)

result7_downsampled = pd.concat([result7_majority_downsampled, result7_minority])


print(result7_downsampled.Deal_status.value_counts())
result7_downsampled.to_csv('Match_train.csv', index = False) 

grid = {"min_child_weight":[1],
            "max_depth":[3,7,9,10],
            "learning_rate":[0.08,0.09,0.1,0.3,0.4],
            "reg_alpha":[0.01,0.02,0.3],
            "reg_lambda":[0.8,0.9,1,1.2,1.4],
            "gamma":[0.002,0.004,0.03],
            "subsample":[0.9,1.0],
            "colsample_bytree":[0.7,0.9,1.0],
            "objective":['binary:logistic'],
            "nthread":[-1],
            "scale_pos_weight":[1],
            "seed":[0,10,42,13],
            "n_estimators": [50,100,200,300,400,500]}

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,
	param_distributions=grid,
	n_iter=5,
	scoring='roc_auc',
	n_jobs=-1,
	cv=5,  #Setting cross-validation folds to 5. It can be changed, however, it would be computationally heavy
	verbose=3)

X=result7_downsampled.iloc[:,0:8]
Y=result7_downsampled.iloc[:,8]
print(Y)

start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,Y)
print(timer(start_time))

best= random_search.best_estimator_
print(best)

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9, gamma=0.03,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=200, n_jobs=1,
              nthread=-1, objective='binary:logistic', random_state=0,
              reg_alpha=0.3, reg_lambda=1.4, scale_pos_weight=1, seed=13,
              silent=None, subsample=1.0, verbosity=1)
print(classifier)

accuracy = []

# Using Stratified K fold CV to get equal classes 
skf = StratifiedKFold(n_splits=10, random_state = None)
skf.get_n_splits(X,Y)

# Fitting the trained classifier model to the training dataset
for train_index, test_index in skf.split(X,Y):
  print("Train",train_index,"Validation", test_index)
  X_train,X_test = X.iloc[train_index], X.iloc[test_index]
  Y_train,Y_test = Y.iloc[train_index], Y.iloc[test_index]

  classifier.fit(X_train,Y_train)
  prediction = classifier.predict(X_test)
  score = accuracy_score(prediction,Y_test)
  accuracy.append(score)

# Printing the Accuracy of the model
print(accuracy)
print(np.array(accuracy).mean())

with open('model_match', 'wb') as f:
	pickle.dump(classifier,f)