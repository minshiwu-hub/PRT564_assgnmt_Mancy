# Import libraries
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Binarizer

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingDriverBehaviourInliersData.csv'

# Define the data variable names
myNames = ['Experience', 'KlmsDist', 'Meter', 'Occupants', 'VehSeq', 'Airbag', 'AlcoholRel', 'LicClass', 'PedVMC',
           'HeavyVeh', 'Carriage', 'DUI', '4WD', 'Intername', 'LicStatus', 'NTRes', 'Community', 'Country', 'SafDevice',
           'Sex', 'State1', 'RegoState', 'RoadDivision', 'RoadName', 'RoadUser1', 'RoadWidth', 'Rural', 'SpeedRel',
           'SurfaceType', 'TSD', 'UnitType', 'VehMov', 'VehVMC', 'VertFeature', 'Weather', 'CyclistInv', 'ArticVeh1',
           'RigidVeh', 'ArticVeh2', 'ContFactor']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Transform data set into an array
myArray = myDataframe.values

# Split the array to input (39 explanatory variables) and output (1 response variable)
myExplanatoryvariables = myArray[:,0:39]
myResponsivevariable = myArray[:,39]

myBinariser = Binarizer(threshold=0.0).fit(myExplanatoryvariables)
myBinarisedata = myBinariser.transform(myExplanatoryvariables)

set_printoptions(precision=3)
print()
print(myBinarisedata[0:5,:])
print()