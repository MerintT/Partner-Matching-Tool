import numpy as np
import pandas as pd
import pickle
import mysql.connector
import configparser

config = configparser.ConfigParser()
config.read("configm.ini")

with open('model_match', 'rb') as f:
	mp = pickle.load(f)

mydb = mysql.connector.connect(
  host=config.get('db-connection','host'),
  user=config.get('db-connection','user'),
  password=config.get('db-connection','passcode'),
  database=config.get('db-connection','name')
)

query = config.get('data-extraction','mainquery') 

df = pd.read_sql(query, mydb)
print(df)

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

result1 = dataMerge(df1=df,df2=df_drop,col='Dropoff_location')
result1.drop(["Dropoff_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result1 = result1.rename(columns={'Shares_accepted': 'Dropoff_location'}) # Renaming columns

result2 = dataMerge(df1=result1,df2=df_pick,col='Pickup_location')
result2.drop(["Pickup_location","Count", "when_accepted"],inplace=True,axis=1) # Removing old pickup location values
result2 = result2.rename(columns={'Shares_accepted': 'Pickup_location'})

result3 = dataMerge(df1=result2,df2=df_requester,col='Requester')
result3.drop(["Requester","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result3 = result3.rename(columns={'Frequency': 'Requester'})# Renaming columns

result4 = dataMerge(df1=result3,df2=df_add,col='Addressee')
result4.drop(["Addressee","Count", "when_accepted"],inplace=True,axis=1)# Removing old pickup location values
result4 = result4.rename(columns={'Frequency': 'Addressee'})# Renaming columns

result4 = result4.fillna(0)

result4.drop(['Requirement_id'], axis=1 , inplace=True)

result4 = pd.concat([result4,pd.get_dummies(result4['Direction'])],axis=1)
result4.drop(['Direction'],axis=1, inplace=True)

#result4 = pd.concat([result4,pd.get_dummies(result4['Container_Type'])],axis=1)
#result4.drop(['Container_Type'],axis=1, inplace=True)

#result5 = result4.head()
#result5.to_csv('Match_final.csv', index = False) 
print(result4)

pred_try = mp.predict(result4)

pred_try_df=pd.DataFrame(pred_try, columns=['Match_Prediction']) 

Req = df.Requirement_id
Req = Req.reset_index(drop=True)
Req = pd.DataFrame(Req,columns=['Requirement_id'])

Company = df.Requester
Company = Company.reset_index(drop=True)
Company = pd.DataFrame(Company,columns=['Requester'])

Partner = df.Addressee
Partner = Partner.reset_index(drop=True)
Partner = pd.DataFrame(Partner,columns=['Addressee'])

Pick = df.Pickup_location
Pick = Pick.reset_index(drop=True)
Pick = pd.DataFrame(Pick,columns=['Pickup_location'])

Drop = df.Dropoff_location
Drop = Drop.reset_index(drop=True)
Drop = pd.DataFrame(Drop,columns=['Dropoff_location'])

final_pred = pd.concat([Req, Company, Partner, Pick, Drop, pred_try_df], axis=1)

final_pred.rename(columns = {'Requirement_id':'requirement_id', 'Requester':'requester_id', 
                              'Addressee':'partner_id', 'Pickup_location':'pickup_id',
                              'Dropoff_location':'dropoff_id', 'Match_Prediction':'match_prediction'}, inplace = True)

print(final_pred)
#final_pred.to_csv('Match_final_result2.csv', index = False) 


from sqlalchemy import create_engine
import pymysql


# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@db-connectivity/{db}"
                       .format(user=config.get('db-push','dbuser'),
                               pw=config.get('db-push','password'),
                               db=config.get('db-push','dbname')))

# Insert whole DataFrame into MySQL
final_pred.to_sql('T_REQUIREMENT_MATCH_PREDICTION', con = engine, if_exists = 'replace', chunksize = 100000, index = False)