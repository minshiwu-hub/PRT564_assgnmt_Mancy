# Import libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:39]
myResponsivevariable = myArray[:,39]
myTestsize = 0.33
myRandomseed = 0

# Split the array into training and test
myExplanatoryvariablestrain, myExplanatoryvariablestest, myResponsivevariabletrain, myResponsivevariabletest = train_test_split (myExplanatoryvariables, myResponsivevariable, test_size = myTestsize, random_state = myRandomseed)

# Setup evaluation algorithm
myModel = LogisticRegression(solver='liblinear', max_iter=10000)
myFitmodel = myModel.fit(myExplanatoryvariablestrain, myResponsivevariabletrain)
myResult = myModel.score(myExplanatoryvariablestest, myResponsivevariabletest)

# Print results
print()
print('Accuracy: %.3f%%' % (myResult*100.0))
print()