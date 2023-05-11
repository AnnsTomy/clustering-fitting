# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:07:35 2023

@author: Anns Tomy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing


def world_data(file_name, col, value1,years):
    # Reading Data for dataframe
    df_climate = pd.read_csv(file_name, skiprows = 4)
    # Grouping data with col value
    df1 =  df_climate.groupby(col, group_keys = True)
    #retriving the data with the all the group element
    df1 = df1.get_group(value1)
    #Reseting the index of the dataframe
    df1 = df1.reset_index()
    #Storing the column data in a variable
    a = df1['Country Name']
    # cropping the data from dataframe
    df1 = df1.iloc[35:150,years]
    #df1 = df1.drop(columns=['Indicator Name', 'Indicator Code'])
    df1.insert(loc=0, column='Country Name', value=a)
    #Dropping the NAN values from dataframe Column wise
    df1= df1.dropna(axis = 0)
    #transposing the index of the dataframe
    df2 = df1.set_index('Country Name').T
    #returning the normal dataframe and transposed dataframe
    return df1, df2

def error_ranges(dataframe, cluster_labels, centroids):
    """
    Calculate the error ranges for each cluster.

    Parameters:
    dataframe (pandas.DataFrame): The normalized dataset.
    cluster_labels (numpy.ndarray): The predicted cluster labels of the data.
    centroids (numpy.ndarray): The centroids of each cluster.

    Returns:
    list: A list of dictionaries containing the error range for each cluster.
    """
    error_ranges = []
    for i in range(len(centroids)):
        cluster_df = dataframe[cluster_labels == i]
        distances = np.sqrt(np.sum((cluster_df - centroids[i])**2, axis=1))
        max_error = np.max(distances)
        min_error = np.min(distances)
        mean_error = np.mean(distances)
        std_error = np.std(distances)
        error_ranges.append({'Cluster': i+1,
                             'Max Error': max_error,
                             'Min Error': min_error,
                             'Mean Error': mean_error,
                             'Std Error': std_error})
    return error_ranges

# countries which are using for data analysis
years= [35,36,37,38,39,40]
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
agri_x, agri_y = world_data("D:\API_19_DS2_en_csv_v2_5346672.csv",
                                       "Indicator Name", "Agricultural land (% of land area)",
                                       years)
#Printing value by countries
print(agri_x)

#Printing value by Year
print(agri_y)

#returns a numpy array as x
x = agri_x.iloc[:,1:].values

def normalizing(value):
    #storing normalization function to min_max_scaler
    min_max_scaler = preprocessing.MinMaxScaler()
    #fitted the array data for normalization
    x_scaled = min_max_scaler.fit_transform(value)
    #Storing values in dataframe named data
    data = pd.DataFrame(x_scaled)
    return data

#caling normalization function
normalized_df = normalizing(x)
print(normalized_df)

def n_cluster(dataFrame,n):
    wcss = []
    for i in range(1, n):
        kmeans = KMeans(n_clusters = i,init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(dataFrame)
        wcss.append(kmeans.inertia_)
    return wcss

k = n_cluster(normalized_df,10)
print(k)

plt.figure(figsize=(15,7))
plt.plot(range(1, 10), k)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster

lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids= kmeans.cluster_centers_
print('centroids=',centroids)

# Call the error_ranges function
error_ranges = error_ranges(normalized_df,lables, centroids)
# Print the error ranges for each cluster
print('Error Ranges:')
for er in error_ranges:
    print(er) 

plt.figure(figsize=(15,7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')

#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')

plt.legend()
# Title of the  plot
plt.title('Clusters of Agricultural land (% of land area) 113 countries for year 1990 to 1994')
plt.xlabel('Countries')
plt.ylabel('Agricultural land (% of land area)')
plt.show()

agri_x['lables']=lables
print('dataframe with cluster lables', agri_x)
agri_x.to_csv('total Agricultural data with cluster label.csv')

years= [35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
prep_x,prep_y = world_data("D:\API_19_DS2_en_csv_v2_5346672.csv",
                         "Indicator Name", "Average precipitation in depth (mm per year)",years)
prep_y['mean']=prep_y.mean(axis=1)
prep_y['years'] = prep_y.index

print(prep_y)

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1999.0
    f = n0 * np.exp(g*t)
    return f

print(type(prep_y["years"].iloc[1]))
prep_y["years"] = pd.to_numeric(prep_y["years"])
print(type(prep_y["years"].iloc[1]))
#calling exponential function
param, covar = opt.curve_fit(exponential, prep_y["years"], prep_y["mean"],
                             p0=(73233967692.102798, 0.03))

prep_y["fit"] = exponential(prep_y["years"], *param)

prep_y.plot("years", ["mean", "fit"],
           title='Data fitting with n0 * np.exp(g*t)',
           figsize=(13, 7))
plt.show()
print(prep_y)


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

#fitting logistic fit
param, covar = opt.curve_fit(logistic, prep_y["years"], prep_y["mean"],
                             p0=(3e12, 0.03, 1990.0), maxfev=5000)

sigma = np.sqrt(np.diag(covar))
igma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
prep_y["logistic function fit"] = logistic(prep_y["years"], *param)
prep_y.plot("years", ["mean", "fit"],
           title='Data fitting with f = n0 / (1 + np.exp(-g*(t - t0)))',
           figsize=(7, 7))
plt.show()

#predicting years
year = np.arange(1960, 2010)
print(year)
forecast = logistic(year, *param)
print('forecast=',forecast)

plt.figure()
plt.plot(prep_y["years"], prep_y["mean"], label="Precipitation")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Precipitation /year")
plt.legend()
plt.title('Prediction of precipitation from 1960 to 2010')
plt.show()