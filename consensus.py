# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 15:15:14 2019

@author: vegac
"""
#$#REMOVE

# =============================================================================
# 
# Krippendorffs Alpha Coefficient: ratio of observed disagreement vs expected disagreement
# Values range from 0 to 1, where 0 is perfect disagreement and 1 is perfect agreement. 
# Krippendorff suggests: “It is customary to require α ≥ .800. Where tentative conclusions are still acceptable, α ≥ .667 is the lowest conceivable limit”
# 
# 
# Algorithm for making pairs;
# 1. check users in vicinity of gps coordinate (only select their information for this)
# 2. check current available data
# 3. determine amount of pairs (find the balance between validation and collection of new information) TODO: determine HOW, information overlap? metric?
# 4. perform Krippendorf's aplha calculcation
# 5. perform 5W analysis
# 6. ???
# 7 PROFIT
# =============================================================================

import pandas as pd
import geopy.distance
import numpy as np

#%% GOLBAL PARAMS
max_dist = 2
centroid = (53.21153278730481, 6.565275192260743)

#$#REMOVE
#loc = (51.1546768, 5.9306645)

#%% DATASETS
### GPS coordinates  dataset ###
#! Needed: GPS coordinates from front-end API (SESSIONID, LAT, LONG)

# For now, some static coordinates

coordinates = [
        {'sessionid':1, 'loc':(53.213126181639126, 6.569566726684571)}, 
        {'sessionid':2, 'loc':(53.218574114526696, 6.565060615539551)}, 
        {'sessionid':3, 'loc':(53.22689886211476, 6.3523292541503915)},
        {'sessionid':4, 'loc':(51.154676888621, 5.9306645039)},
        {'sessionid':5, 'loc':(53.2104898, 6.537410200000001)},
        {'sessionid':6, 'loc':(53.2247811573688, 6.512915451886215)},
        {'sessionid':7, 'loc':(53.2111450641684, 6.572983035411198)},
        {'sessionid':8, 'loc':(53.20692995531333, 6.560151347483952)},
        {'sessionid':9, 'loc':(51.232342323, 5.24342342)},
        {'sessionid':10, 'loc':(53.1546768, 6.23456768)},
        {'sessionid':11, 'loc':(56.1546768, 5.222266646)},
        {'sessionid':12, 'loc':(54.1546768, 6.2222223466)},
        {'sessionid':13, 'loc':(53.2159251126517, 6.546440896775039)},
        {'sessionid':14, 'loc':(53.20951152616761,6.537490682046041)},
        {'sessionid':15, 'loc':(53.223439528478686, 6.540451840798482)},
        {'sessionid':16, 'loc':(53.23768266941586, 6.554666384766506)},
        {'sessionid':17, 'loc':(53.24415472391833, 6.5840633955697285)},
        {'sessionid':18, 'loc':(53.13725940374463, 6.37154661090176)},
        {'sessionid':19, 'loc':(53.29942380012853, 6.405929681519865)},
        {'sessionid':20, 'loc':(53.20522601378343, 6.551430063197472)},
        {'sessionid':21, 'loc':(53.20944129024872, 6.562475216348275)}
        ]
#$#REMOVE
other_data = [{'id':1, 'data':[]}]

#%% GPS DISTANCE CHECK
# Create current list of relevant data points based on distance

#! Needed: API GET method for append to the tuple

# This function calculates distance between two GPS points (LAT, LON)


class Points:
    def __init__(self, coordinates, centroid):
        self.all_points = coordinates #GET?
        self.centroid = centroid
  
    def calc_dist(self, centroid, loc):
        try:
            dist = round(geopy.distance.distance(centroid, loc).km,2)
            return dist
        except:
            if isinstance(loc, None):
                print("emptyness...")
            else:
                print('invalid data type, should be 2 tuples with (lat, lon)')
                    
    def get_distances(self):
        locs = []
        ids = []
        for i in self.all_points:
            for x in i.values():
                if isinstance(x, tuple):
                    locs.append(x) 
                else:
                    ids.append(x)
        distances_list = []
        for i in locs:
            calced_dist = self.calc_dist(self.centroid, i)
            distances_list.append(calced_dist)    
        output = dict(zip(ids, distances_list))
        return(output)

     


def get_relevant_points():
    d = Points(coordinates, centroid)
    distances = d.get_distances()
    # Use dictionary comprehension to select items < maximum distance
    b = { key: value for key, value in distances.items() if value < max_dist }
    return b

get_relevant_points()
    
   
    

#%% DATA sets

# =============================================================================
# processing the dataset in order to create sets that can be validated;
# 1. retrieve session id's and tweet id's, also answers (later perhaps info_id's?)
# 2. filter on relevant points (sessionids)
# 3. group by tweet id and answer
# 4. calculate KA (per tweet_id/information_id) and return reliability score
# =============================================================================

#$#REMOVE
tweets = pd.read_csv('forxi.csv', delimiter=',', nrows=3) # GET

#Filter 
#! todo: add answers (generate some for now)
tweets_filt = tweets[['id']]

users = list(range(1,len(coordinates)+1))

#randomly generate answers for now (50/50), add colnames manually copy a row when adding more tweets.... #$#REMOVE
answers1 = list(np.random.randint(2, size=len(coordinates)))
answers2 = list(np.random.randint(2, size=len(coordinates)))
answers3 = list(np.random.randint(2, size=len(coordinates)))

#Create dataframe
data = pd.DataFrame(np.column_stack([users, answers1, answers2, answers3]), columns=['sessionid', 'answers1', 'answers2','answers3'])

# assign tweet id's
names = tweets_filt['id'].tolist()
names = list(map(str, names))
    

data.columns = ['sessionid'] + names

# Filter on relevant sessionids
data = data.loc[data['sessionid'].isin(list(get_relevant_points().keys()))]


#%% Krippendorff Alpha
import krippendorff

krippendorff.alpha(data)



