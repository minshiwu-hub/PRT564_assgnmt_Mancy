# Import libraries
from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load the csv file using read_csv function of pandas library
myFilename = 'TrainingRoadCrashInliersData.csv'

# Define the data variable names
myNames = ['Mcycle', 'NbrTow', 'Airbag', 'VehVMC1', 'LicStatus', 'ABSSA3', 'Aboriginal', 'AgeBand', 'SafDevice',
           'Sex', 'State1', 'State2', 'RoadUser1', 'RoadWidth', 'TrafDensity', 'ATVInv', 'UnitType', 'VehDirTravel',
           'VehMov', 'VertFeature', 'RigidVeh', 'Businv', 'Pedestrian', 'InjuryDesc']

# Read dataset
myDataframe = read_csv(myFilename, names=myNames)

# Extract the dataframe values
myArray = myDataframe.values

# Split the array into input and output
myExplanatoryvariables = myArray[:,0:23]
myResponsivevariable = myArray[:,23]
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