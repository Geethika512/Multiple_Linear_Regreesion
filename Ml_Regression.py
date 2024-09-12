'''
In this file we are going to develop multiple linear regression with oops concepts of usind car_purchasing_Data
'''
'''
In this file we are going to develop and Multiple Linear Regression with Oops concept
'''
import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
class TRAINING:
    def __init__(self,purchase):
        try:
            self.df = pd.read_csv(purchase , encoding='ISO-8859-1')
            self.df = self.df.drop(['Customer Name', 'Customer e-mail', 'Country'] , axis=1)
            self.X = self.df.iloc[: , :-1] # independent
            self.y = self.df.iloc[: , -1] # dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            print(f'training data taken dimensions are {(len(self.X_train)) , (len(self.y_train))}')
            print(f'testing data taken dimensions are {(len(self.X_test)) , (len(self.y_test))}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def data_training(self):
        try:
            self.reg = LinearRegression()
            self.reg.fit(self.X_train,self.y_train) # trainnig
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def train_performance(self):
        try:
            self.y_train_pred = self.reg.predict(self.X_train)
            print(f'Train Accuracy : -> {r2_score(self.y_train,self.y_train_pred)}')
            # Mean Squared Error (MSE)
            train_mse = mean_squared_error(self.y_train, self.y_train_pred)
            print(f'Train Mean Squared Error (MSE): {train_mse}')
            # Root Mean Squared Error (RMSE)
            train_rmse = np.sqrt(train_mse)
            print(f'Train Root Mean Squared Error (RMSE): {train_rmse}')
            # Absolute Mean Squared Error (AMSE)
            absolute_diff_squared = np.mean(np.square(np.abs(self.y_train - self.y_train_pred)))
            print(f'Train Absolute Mean Squared Error (AMSE): {absolute_diff_squared}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')
    def testing(self):
        try:
            self.y_test_pred = self.reg.predict(self.X_test)
            print(f'Test Accuracy : -> {r2_score(self.y_test, self.y_test_pred)}')
            # Mean Squared Error (MSE)
            train_mse = mean_squared_error(self.y_train, self.y_train_pred)
            print(f'Test Mean Squared Error (MSE): {train_mse}')
            # Root Mean Squared Error (RMSE)
            train_rmse = np.sqrt(train_mse)
            print(f'Test Root Mean Squared Error (RMSE): {train_rmse}')
            # Absolute Mean Squared Error (AMSE)
            absolute_diff_squared = np.mean(np.square(np.abs(self.y_train - self.y_train_pred)))
            print(f'Test Absolute Mean Squared Error (AMSE): {absolute_diff_squared}')
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')

if __name__ == "__main__":
    try:
        obj = TRAINING('c:\\Users\\Hp\\Documents\\GEETHU\\geethika_ds_course\\pycharm_ide_codes\\MLR\\Car_Purchasing_data.csv') # constructor will be called
        obj.data_training()
        obj.train_performance()
        obj.testing()
    except Exception as e:
        error_type,error_msg,err_line = sys.exc_info()
        print(f'Error from Line {err_line.tb_lineno} -> type {error_type} -> Error msg -> {error_msg}')