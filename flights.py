import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

column_names = ['YEAR','MONTH','DAY','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','DISTANCE','WHEELS_ON','TAXI_IN','SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY','DIVERTED','CANCELLED','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']

airlines = pd.read_csv('./data/airlines.csv')
abbr_companies = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()
airports= pd.read_csv('./data/airports.csv')
flights_v1 = pd.read_csv('./data/flights-small.csv', low_memory=False, names=column_names)

#flights_v1 = pd.merge(flights, airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
#flights_v1.drop('IATA_CODE', axis=1, inplace=True)
#flights_v1.rename(columns={'AIRLINE_x': 'AIRLINE_CODE','AIRLINE_y': 'AIRLINE'}, inplace=True)

#_________________________________________________________
# Function that convert the 'HHMM' string to datetime.time
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400: chaine = 0
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
#_____________________________________________________________________
# Function that combines a date and time to produce a datetime.datetime
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
#_______________________________________________________________________________
# Function that combine two columns of the dataframe to create a datetime format
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)

flights_v1['DATE'] = pd.to_datetime(flights_v1[['YEAR','MONTH', 'DAY']])
flights_v1['SCHEDULED_DEPARTURE'] = create_flight_time(flights_v1, 'SCHEDULED_DEPARTURE')
flights_v1['DEPARTURE_TIME'] = flights_v1['DEPARTURE_TIME'].apply(format_heure)
flights_v1['SCHEDULED_ARRIVAL'] = flights_v1['SCHEDULED_ARRIVAL'].apply(format_heure)
flights_v1['ARRIVAL_TIME'] = flights_v1['ARRIVAL_TIME'].apply(format_heure)
days = {1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday',7:'Sunday'}
months = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
flights_v1['DAY_OF_WEEK'] = flights_v1['DAY_OF_WEEK'].apply(lambda x: days[x])
flights_v1['MONTH'] = flights_v1['MONTH'].apply(lambda x: months[x])

variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF', 'YEAR', 
                       'DAY','DATE', 'AIR_SYSTEM_DELAY',
                       'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY',
                       'WEATHER_DELAY', 'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
                       'FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME', 'SCHEDULED_DEPARTURE', 
                       'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 
                       'SCHEDULED_TIME', 'ELAPSED_TIME', 'ARRIVAL_DELAY']
flights_v1.drop(variables_to_remove, axis = 1, inplace = True)
flights_v1 = flights_v1[['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','MONTH','DAY_OF_WEEK','DEPARTURE_DELAY']]

# show sample 5 rows
print(flights_v1.head(5))

flights_shuffled = shuffle(flights_v1)

train, test = train_test_split(flights_shuffled, test_size=0.2)

print("Training dataset size: " + str(len(train)))
print("Test dataset size: " + str(len(test)))

def df_to_dataset(dataframe, shuffle=True):
  dataframe = dataframe.copy()
  labels = dataframe.pop('DEPARTURE_DELAY')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  return ds

train_ds = df_to_dataset(train, shuffle=False)
test_ds = df_to_dataset(test, shuffle=False)

