# Input libraries
from pandas import read_csv

# Load the csv file using read_csv function of pandas library
filename = 'RoadCrashData.csv'
names = ['Month', 'WeekDay','DaySpan', 'Light', 'Weather', 'Traffic', 'Surface', 'Division', 'TowFactor',
            'HeavyVehicle', 'LGA/Area', 'Rural/Urban', 'SpeedRelated', 'RoadClass']

# Read the csv file
data = read_csv(filename, names=names)

# Print the data types of attributes of the dataset
type = data.dtypes
print(type)