import numpy as np
 
X = np.arange(-25, 25, 1).reshape(10,5)
 
ones = np.ones((X.shape[0], 1))
 
X_1 = np.append(X.copy(), ones, axis=1)
 
w = np.random.rand(X_1.shape[1])
 
 
def predict(x, w):     
    total_stimulation = np.dot(x, w)       
    y_pred = 1 if total_stimulation > 0 else -1
    return y_pred

print(w)

def predictAll(matrix, w):
    for m in matrix:
       predicted = predict(m, w)
       print("###############")
       print(m)
       print('predicted => ', predicted)
        
predictAll(X_1, w)
        