

                   # Ładowanie bibliotek
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
 
# Ładowanie danych
auto = pd.read_csv(r"/Users/darekdajcz/Desktop/Uczenie_maszynowe_Python/Kurs ML/auto-mpg.csv")
auto.head()
auto.shape
 
# Przygotowanie danych
X = auto.iloc[:, 1:-1]
X = X.drop('horsepower', axis=1)
y = auto.loc[:,'mpg']
 
X.head()
y.head()
 
# Budowanie modelu
lr =  LinearRegression()
lr.fit(X.to_numpy(),y)
lr.score(X.to_numpy(),y)
 
# Korzystanie z modelu
my_car1 = [4, 160, 190, 12, 90, 1]
my_car2 = [4, 200, 260, 15, 83, 1]
cars = [my_car1, my_car2]
 
mpg_predict = lr.predict(cars)
print(mpg_predict)