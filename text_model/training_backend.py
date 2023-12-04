import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import pickle

def train_models(model,x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=5)
    model.fit(x_train,y_train)
    pre = model.predict(x_test)
    r2 = r2_score(y_test,pre)
    rmse = np.sqrt(mean_squared_error(y_test,pre))
    return r2, rmse

def best_model(r2,rmse):
    max_score = r2[0]
    for i in range(len(r2)):
        if r2[i] > max_score:
            max_score = r2[i]
            best = i
            avg_error = rmse[i]
    return best, max_score, avg_error

def train_best_model(index,model_list,x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=5)
    model_list[index].fit(x_train,y_train)
    pickle.dump(model_list[index], open('RFmodel.pkl', 'wb'))
