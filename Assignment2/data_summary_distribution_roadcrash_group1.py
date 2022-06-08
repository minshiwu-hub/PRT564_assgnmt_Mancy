# Import libraries
from pandas import read_csv

# Load the csv file using read_csv function of pandas library
myFilename = 'RoadCrashData.csv'
myNames = ['Month', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division', 'TowFactor',
            'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# Read the csv file
myData = read_csv(myFilename, names=myNames)

# Count clases to view imbalance in the dataset which would introduce bias in the model
myClass_counts = myData.groupby('RoadClass').size()

# Print the class count
print(myClass_counts)
