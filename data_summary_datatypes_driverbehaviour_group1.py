# Input libraries
from pandas import read_csv

# Load the csv file using read_csv function of pandas library
myFilename = 'DriverBehaviourData.csv'
myNames = ['Occupants', 'Age', 'Sex', 'Aboriginal', 'Day', 'Alcohol', 'Drugs', 'Circumstance', 'RUMDesc', 
            'Pedestrian', 'Cyclist', 'DriverClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# print the data types of attributes of the dataset
myType = myData.dtypes
print(myType)