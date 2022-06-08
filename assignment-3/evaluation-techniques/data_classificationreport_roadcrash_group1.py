# Import libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Split the array into training and test
myExplanatoryvariablestrain, myExplanatoryvariablestest, myResponsivevariabletrain, myResponsivevariabletest = train_test_split (myExplanatoryvariables, myResponsivevariable, test_size = myTestsize, random_state = myRandomseed)

# Setup predictive algorithm
myModel = LogisticRegression(solver='liblinear', max_iter=10000)
myFitmodel = myModel.fit(myExplanatoryvariablestrain, myResponsivevariabletrain)
myPredictedmodel = myFitmodel.predict(myExplanatoryvariablestest)

# Print results
myReport = classification_report(myResponsivevariabletest, myPredictedmodel)
print()
print(myReport)
print()