from sklearn.preprocessing import StandardScaler
import pandas
import numpy

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

print(numpy.shape(X),numpy.shape(Y))
np = numpy.array(X)
print(np[0:4,:-2])

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# print Y
#for i in range(len(X)):
#    for j in range(len(X[i])):
#        print(X[i][j], end=' ')
#    print()

# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
rescaledXframe = pandas.DataFrame(rescaledX)
print(rescaledXframe.describe())