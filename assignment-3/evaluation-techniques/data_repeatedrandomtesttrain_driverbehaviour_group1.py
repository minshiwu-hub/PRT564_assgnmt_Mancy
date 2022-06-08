# Import libraries
from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
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

# Setup evaluation algorithm
myNsplits = 10
myValidation = ShuffleSplit(n_splits=myNsplits, test_size=myTestsize, random_state = myRandomseed)
myModel = LogisticRegression(solver='liblinear', max_iter=10000)
myResult = cross_val_score(myModel, myExplanatoryvariables, myResponsivevariable, cv=myValidation)

# Print results
print()
print('Accuracy: %.3f%% (%.3f%%) ' % (myResult.mean()*100.0, myResult.std()*100.0))
print()