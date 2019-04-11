import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras import layers
import pandas as pd

column_names = ['YEAR','MONTH','DAY','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER','TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','DISTANCE','WHEELS_ON','TAXI_IN','SCHEDULED_ARRIVAL','ARRIVAL_TIME','ARRIVAL_DELAY','DIVERTED','CANCELLED','CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
dataset = pd.read_csv('./data/flights-small.csv', delimiter = ',', names=column_names)

#print(dataset.head(4))
#print(dataset.sample(5))
#print(dataset[['AIRLINE','FLIGHT_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','DEPARTURE_TIME']])

airlines = pd.read_csv('./data/airlines.csv')
airports= pd.read_csv('./data/airports.csv')
flights = pd.read_csv('./data/flights-small.csv', low_memory=False, names=column_names)

flights_v1 = pd.merge(flights, airlines, left_on='AIRLINE', right_on='IATA_CODE', how='left')
flights_v1.drop('IATA_CODE', axis=1, inplace=True)
flights_v1.rename(columns={'AIRLINE_x': 'AIRLINE_CODE','AIRLINE_y': 'AIRLINE'}, inplace=True)

airport_mean_delays = pd.DataFrame(pd.Series(flights['ORIGIN_AIRPORT'].unique()))
airport_mean_delays.set_index(0, drop = True, inplace = True)
abbr_companies = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()
identify_airport = airports.set_index('IATA_CODE')['CITY'].to_dict()

# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
#___________________________________________________________

for carrier in abbr_companies.keys():
    fg1 = flights[flights['AIRLINE'] == carrier]
    test = fg1['DEPARTURE_DELAY'].groupby(flights['ORIGIN_AIRPORT']).apply(get_stats).unstack()
    airport_mean_delays[carrier] = test.loc[:, 'mean'] 

airline_rank_v09 = pd.DataFrame(flights_v1.groupby(['AIRLINE'])['AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'].sum()).reset_index()
airline_rank_v09['total'] = airline_rank_v09['AIR_SYSTEM_DELAY'] + airline_rank_v09['AIRLINE_DELAY'] + airline_rank_v09['LATE_AIRCRAFT_DELAY'] + airline_rank_v09['WEATHER_DELAY']
airline_rank_v09['pcnt_LATE_AIRCRAFT_DELAY'] = (airline_rank_v09['LATE_AIRCRAFT_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_AIRLINE_DELAY'] = (airline_rank_v09['AIRLINE_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_AIR_SYSTEM_DELAY'] = (airline_rank_v09['AIR_SYSTEM_DELAY']/airline_rank_v09['total'])
airline_rank_v09['pcnt_WEATHER_DELAY'] = (airline_rank_v09['WEATHER_DELAY']/airline_rank_v09['total'])

airline_rank_v01 = pd.DataFrame({'flight_volume' : flights_v1.groupby(['AIRLINE'])['FLIGHT_NUMBER'].count()}).reset_index()
airline_rank_v01.sort_values("flight_volume", ascending=True, inplace=True)
flight_volume_total = airline_rank_v01['flight_volume'].sum()
airline_rank_v01['flight_pcnt'] = airline_rank_v01['flight_volume']/flight_volume_total

airline_rank_v02 = pd.DataFrame({'cancellation_rate' : flights_v1.groupby(['AIRLINE'])['CANCELLED'].mean()}).reset_index()
airline_rank_v02.sort_values("cancellation_rate", ascending=False, inplace=True)
airline_rank_v03 = pd.DataFrame({'divertion_rate' : flights_v1.groupby(['AIRLINE'])['DIVERTED'].mean()}).reset_index()
airline_rank_v03.sort_values("divertion_rate", ascending=False, inplace=True)
airline_rank_v1 = pd.merge(airline_rank_v01, airline_rank_v02, left_on='AIRLINE', right_on='AIRLINE', how='left')
airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v03, left_on='AIRLINE', right_on='AIRLINE', how='left')

airline_rank_v07 = pd.DataFrame({'avg_arrival_delay' : flights_v1.groupby(['AIRLINE'])['ARRIVAL_DELAY'].mean()}).reset_index()
airline_rank_v08 = pd.DataFrame({'avg_departure_delay' : flights_v1.groupby(['AIRLINE'])['DEPARTURE_DELAY'].mean()}).reset_index()
airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v07, left_on='AIRLINE', right_on='AIRLINE', how='left')
airline_rank_v1 = pd.merge(airline_rank_v1, airline_rank_v08, left_on='AIRLINE', right_on='AIRLINE', how='left')

print(airline_rank_v1.head(25))


day_of_week_vocab = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" ]
day_of_week_column = tf.feature_column.categorical_column_with_vocabulary_list(
      key="day_of_week", vocabulary_list=day_of_week_vocab)

airline_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="airline",
        vocabulary_file="./data/airlines.txt",
        vocabulary_size=14)

airport_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="airport",
        vocabulary_file="./data/airports.txt",
        vocabulary_size=322)

feature_columns = [ day_of_week_column, airline_feature_column, airport_feature_column ]


airline_one_hot = pd.get_dummies(flights_v1['ORIGIN_AIRPORT'])

print(airline_one_hot.info())
print("First row values: " + str(airline_one_hot.values[0]))

# build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_ds.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)

model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

model.summary()

EPOCHS = 10

# setup TB callback
tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='/chzbrgr71/flight-delays/logs')

model.fit(train_ds, epochs=EPOCHS, validation_split = 0.2, verbose=0, callbacks=[tensorboard_cbk])