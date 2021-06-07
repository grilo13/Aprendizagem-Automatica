#Importacao das bibliotecas necessárias
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

read = pd.read_csv("weather.nominal.csv")

codificar_X = LabelEncoder()
read = read.apply(LabelEncoder().fit_transform)

X = read.drop('play', axis=1)
y = read['play']

#Carrega e prepara os dados do ficheiro
data=np.genfromtxt("weather.nominal.csv", delimiter=",", dtype=None, encoding=None)
xdata = X
ydata = y

classifier = DecisionTreeClassifier(criterion='gini')

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, random_state=0, test_size=0.75)

classifier.fit(x_train, y_train)

result = classifier.score(x_test, y_test)

print("\n----Decision Tree based in sklearn with DecisionTreeClassifier----\n")

print("Percentagem de casos corretamente classificados {:2.2%}".format(result))