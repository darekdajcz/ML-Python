  
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
 

auto = pd.read_csv(r"/Users/darekdajcz/Desktop/Uczenie_maszynowe_Python/Kurs_ML/iris_diagrams/csv_data/auto-mpg.csv")
auto.head()
auto.shape
 
X = auto.iloc[:, 1:-1]
X = X.drop('horsepower', axis=1)
y = auto.loc[:,'mpg']
 
X.head()
y.head()
 
lr =  LinearRegression()
fit = lr.fit(X.to_numpy(),y)
score = lr.score(X.to_numpy(),y)
 
my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]

cars = [my_car1, my_car2]
 
mpg_predict = lr.predict(cars)
print(mpg_predict)
