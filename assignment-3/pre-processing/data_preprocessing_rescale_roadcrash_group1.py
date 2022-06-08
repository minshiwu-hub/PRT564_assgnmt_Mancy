# Import libraries
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingRoadCrashInliersData.csv'

# Define the data variable names
myNames = ['Mcycle', 'NbrTow', 'Airbag', 'VehVMC1', 'LicStatus', 'ABSSA3', 'Aboriginal', 'AgeBand', 'SafDevice',
           'Sex', 'State1', 'State2', 'RoadUser1', 'RoadWidth', 'TrafDensity', 'ATVInv', 'UnitType', 'VehDirTravel',
           'VehMov', 'VertFeature', 'RigidVeh', 'Businv', 'Pedestrian', 'InjuryDesc']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Transform data set into an array
myArray = myDataframe.values

# Split the array to input (23 explanatory variables) and output (1 response variable)
myExplanatoryvariables = myArray[:,0:23]
MyResponsivevariable = myArray[:,23]

myScaler = MinMaxScaler(feature_range=(0,1))
myStandardisedata = myScaler.fit_transform(myExplanatoryvariables)

set_printoptions(precision=3)
print()
print(myStandardisedata[0:5,:])
print()