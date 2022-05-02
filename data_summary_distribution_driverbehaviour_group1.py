# Import libraries
from pandas import read_csv

# Load the csv file using read_csv function of pandas library
myFilename = 'DriverBehaviourData.csv'
myNames = ['Occupants', 'Age', 'Sex', 'Aboriginal', 'Day', 'Alcohol', 'Drugs', 'Circumstance', 'RUMDesc', 
            'Pedestrian', 'Cyclist', 'DriverClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Count clases to view imbalance in the dataset which would introduce bias in the model
myClass_counts = myData.groupby('DriverClass').size()

# Print the class count
print(myClass_counts)