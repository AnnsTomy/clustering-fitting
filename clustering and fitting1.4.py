# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:19:00 2023

@author: Anns Tomy
"""
#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import scipy.optimize as opt


#Define the function to read the CSV file
def world_data(file_name, col, value1, years):
    """
    Reads a CSV file containing World Bank data, extracts the data, and returns two dataframes - one with the extracted data and another
    with the extracted data transposed.

    Parameters:
    -----------
    file_name: str
        The name of the CSV file containing the World Bank Climate data.
    col: str
        The column name on which to group the data (e.g. 'Country Name' or 'Region').
    value1: str
        The value of the column on which to filter the data (e.g. 'United States' or 'Europe & Central Asia').
    years: list of int
        A list of integers representing the years for which to extract data (e.g. [2000, 2001, 2002]).

    Returns:
    --------
    A tuple containing two dataframes:
        - df1: A dataframe with the extracted data for the given country/region and years.
        - df2: A transposed dataframe with the extracted data for the given country/region and years.

    """
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
        cluster_df = dataframe[cluster_labels == i]  # Select the dataframe for the current cluster
        distances = np.sqrt(np.sum((cluster_df - centroids[i])**2, axis=1)) # Calculate the distances from each point in the cluster to the centroid
        max_error = np.max(distances) # Calculate the max, min, mean, and standard deviation of the distances
        min_error = np.min(distances)
        mean_error = np.mean(distances)
        std_error = np.std(distances)
        error_ranges.append({'Cluster': i+1,     # Store the error range information in a dictionary
                             'Max Error': max_error,
                             'Min Error': min_error,
                             'Mean Error': mean_error,
                             'Std Error': std_error})
    return error_ranges


def normalizing(value):
    """
    Normalize the input data using Min-Max Scaler.

    Parameters:
    -----------
    value: numpy array or pandas DataFrame
        Input data to be normalized

    Returns:
    --------
    pandas DataFrame
        Normalized data in a DataFrame format
    """
    min_max_scaler = preprocessing.MinMaxScaler() # initialize the Min-Max Scaler
    x_scaled = min_max_scaler.fit_transform(value) # fit and transform the input data to the scaler
    data = pd.DataFrame(x_scaled)
    return data # convert the normalized data to a DataFrame and return it


def n_cluster(dataFrame, n):
    """
    Computes the Within-Cluster-Sum-of-Squares (WCSS) for a range of cluster sizes, given a dataFrame and a maximum number of clusters.

    Args:
        dataFrame (pd.DataFrame): The data frame to be used for clustering.
        n (int): The maximum number of clusters to be evaluated.

    Returns:
        List[float]: A list of WCSS values for each evaluated number of clusters. The list has length `n - 1`."""
# Create an empty list to store the WCSS values
    wcss = []
    for i in range(1, n): # Loop through the range of number of clusters from 1 to n
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0) # Create a KMeans object with i number of clusters, and fit the data to it
        kmeans.fit(dataFrame) 
        wcss.append(kmeans.inertia_) # Append the WCSS value for the fitted KMeans object to the list
    return wcss # Return the list of WCSS values


# countries which are using for data analysis
years= [35, 36, 37, 38, 39, 40] #select the years
'''calling dataFrame functions for all the dataframe which will be used for visualization'''
agri_x, agri_y = world_data("D:\API_19_DS2_en_csv_v2_5346672.csv",
                                       "Indicator Name", "Agricultural land (% of land area)",
                                       years) #create the dataframe
#Printing value by countries
print(agri_x)

#Printing value by Year
print(agri_y)

#returns a numpy array as x
x = agri_x.iloc[:,1:].values

#caling normalization function
normalized_df = normalizing(x)

#print the normalised_df
print(normalized_df)

#determine the k value
k = n_cluster(normalized_df,10)

#print k value
print(k)

# Plot the WCSS for different number of clusters to find the elbow point
plt.figure(figsize = (15,7)) #set figure size
plt.plot(range(1, 10), k) #set the range
plt.title('The elbow method') #set title
plt.xlabel('Number of clusters') #set x label
plt.ylabel('WCSS')  #set y label
plt.show() 

#finding k means cluster
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100,
                n_init = 10, random_state = 0)

#fitting and predicting the data using k means cluster
lables = kmeans.fit_predict(normalized_df)

#finding Centroids for Kmean Cluster
centroids = kmeans.cluster_centers_

#print the centroids
print('centroids=', centroids)

# Call the error_ranges function
error_ranges = error_ranges(normalized_df, lables, centroids)

# Print the error ranges for each cluster
print('Error Ranges:')
for er in error_ranges:
    print(er) 

#ploting the clustering of the normalized.df    
plt.figure(figsize = (15, 7))
#Ploting cluster 1
plt.scatter(normalized_df.values[lables == 0, 0], normalized_df.values[lables == 0, 1], s = 100, c = 'green', label = 'Cluster1')
#Ploting cluster 2
plt.scatter(normalized_df.values[lables == 1, 0], normalized_df.values[lables == 1, 1], s = 100, c = 'orange', label = 'Cluster2')
#Ploting cluster 3
plt.scatter(normalized_df.values[lables == 2, 0], normalized_df.values[lables == 2, 1], s = 100, c = '#EEC591', label = 'Cluster3')
#Ploting centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label = 'Centroids')
#Add the legends
plt.legend()
# Title of the  plot
plt.title('Clusters of Agricultural land (% of land area) 113 countries for year 1990 to 1994')
#set x label
plt.xlabel('Countries')
#set y label
plt.ylabel('Agricultural land (% of land area)')
plt.show()

#converting agri_x to csv
agri_x['lables'] =lables
print('dataframe with cluster lables', agri_x) 
agri_x.to_csv('total Agricultural data with cluster label.csv')

#creating a new dataframe
years = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
prep_x, prep_y = world_data("D:\API_19_DS2_en_csv_v2_5346672.csv",
                         "Indicator Name", "Average precipitation in depth (mm per year)",years)
prep_y['mean'] = prep_y.mean(axis = 1) #finding mean
prep_y['years'] = prep_y.index #making the index as years

#print the prep_y
print(prep_y)

#Plotting the mean agricultural land of different countries
ax = prep_y.plot(x = 'years', y = 'mean', figsize=(13, 7), title = 'Mean Agricultural land of different countries over the given years', xlabel = 'Years', ylabel = 'rate of agricultural land (% of area)')


#define the exponenetial function
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1999.0 #shift the time values to start at 0
    f = n0 * np.exp(g*t)#calculate the exponential function
    return f
   

#Convert the "years" column of the "prep_y" DataFrame to numeric
prep_y["years"] = pd.to_numeric(prep_y["years"])
#Check the data type of the "years" column
print(type(prep_y["years"].iloc[1]))
#Fit the data to the exponential function
param, covar = opt.curve_fit(exponential, prep_y["years"], prep_y["mean"],
                             p0 = (73233967692.102798, 0.03))
#Calculate the fitted values and store them in a new column "fit"
prep_y["fit"] = exponential(prep_y["years"], *param)


#Define the logistic function
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


#fitting logistic fit
param, covar = opt.curve_fit(logistic, prep_y["years"], prep_y["mean"],
                             p0=(3e12, 0.03, 1990.0), maxfev=5000)
sigma = np.sqrt(np.diag(covar))

#Printing parameters and standard deviation
print("parameters:", param)
print("std. dev.", sigma)

#Creating new column with fitted values
prep_y["logistic function fit"] = logistic(prep_y["years"], *param)
param, covar = opt.curve_fit(logistic, prep_y["years"], prep_y["mean"],
                             p0=(3e12, 0.03, 1990.0), maxfev=5000)
sigma = np.sqrt(np.diag(covar))

#print the parameters standard deviation
print("parameters:", param)
print("std. dev.", sigma)
prep_y["logistic function fit"] = logistic(prep_y["years"], *param)

# scatter plot of the data
plt.scatter(prep_y["years"], prep_y["mean"], label="Data")
# plot the logistic function fit
plt.plot(prep_y["years"], prep_y["logistic function fit"], label="Logistic fit")
# set labels and legend
plt.xlabel("Years")
plt.ylabel("Mean")
plt.title("Logistic Function fit of mean of Precipitation Data")
plt.legend()
# display the plot
plt.savefig('logisticsfit.png')
plt.show()

#predicting years
year = np.arange(1960, 2010)
print(year)
forecast = logistic(year, *param)
print('forecast=',forecast)
#Plotting original data and forecast
plt.figure()
plt.plot(prep_y["years"], prep_y["mean"], label="Precipitation")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Precipitation /year")
plt.legend()
plt.title('Prediction of precipitation from 1960 to 2010')
plt.show()