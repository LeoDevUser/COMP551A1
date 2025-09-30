import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy import stats
from sklearn.preprocessing import StandardScaler
matplotlib.use('Qt5Agg') #Set GUI backend for plots

#parse arguments
def check_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")

    if ivalue <= 0 or ivalue > 100:
        raise argparse.ArgumentTypeError(f"'{value}' must be an integer between 0 and 100 (exclusive).")

    return ivalue
#create parser
parser = argparse.ArgumentParser(usage='linear.py [--split] [values]')
#add args
parser.add_argument('--split', type=check_int, help='Represents the fraction used as the training set', default=80)
parser.add_argument('--true', action='store_true', help='When set drops total_UPDRS from the features', default=False)
args = parser.parse_args()
if args.split == 100:
    args.split = 99

# fetch dataset
df = pd.read_csv('parkinsons_updrs.csv')
pd.set_option('display.max_columns', None)

# data (as pandas dataframes)
X = df.drop(columns=['motor_UPDRS', 'subject#']) #Features
if (args.true):
    X = X.drop(columns = ['total_UPDRS'])#Drop the target from the features
Y = df[['motor_UPDRS']] #Target
print(Y.std())

#get correlation matrix
corr = abs(X.corr())
#get upper triangular matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
# find features with correlation greater than 0.9 with each other 
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
# drop highly correlated features
X = X.drop(to_drop, axis=1)

#drop outliers with z score greater than 3 
z = np.abs(stats.zscore(X))
#~0.2% (0.1 above and 0.1 below the mean) of data in a normally distributed dataset will fall in this range
outlier_indices = np.where(z > 3)[0]
X = X.drop(outlier_indices)
Y = Y.drop(outlier_indices)

class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def fit(self,x,y):
        if x.ndim == 1:
            x = x[:,None]
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        self.w = np.linalg.lstsq(x,y)[0]
        return self

    def predict(self,x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return yh

#Split Data
train_range = int(args.split / 100 * len(X))
X_train = X.iloc[:train_range]
X_predict = X.iloc[train_range:]
Y_train = Y.iloc[:train_range]
Y_predict = Y.iloc[train_range:]

#Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_predict = scaler.transform(X_predict)

#Fit and Predict
model = LinearRegression().fit(X_train,Y_train)
yh_train = model.predict(X_train)
yh_test = model.predict(X_predict)

#Cost Function for prediction
mse_test = np.mean((Y_predict - yh_test)**2)
rmse_test = np.sqrt(mse_test)
mse_train = np.mean((Y_train - yh_train)**2)
rmse_train = np.sqrt(mse_train)

def R2(y,yh):
    y = y.values
    y_mean = np.mean(y)
    ss_res = np.sum((y-yh)**2)
    ss_tot = np.sum((y-y_mean)**2)
    return 1 - ss_res / ss_tot

print(f'Results for a {args.split}/{100-args.split} train/test split:')
print(f'\nFeature names and their corresponding weights:')
for i, col in enumerate(X.columns):
    print(f"{col}: {model.w[i][0]}")
print('\nResults for train set:')
print(f"MSE: {mse_train}")
print(f"RMSE: {rmse_train}")
print(f"R^2: {R2(Y_train, yh_train)}")
print('\nResults for test set:')
print(f"MSE: {mse_test}")
print(f"RMSE: {rmse_test}")
print(f"R^2: {R2(Y_predict, yh_test)}")

plt.plot(Y_predict,yh_test,'.')
plt.plot([Y_predict.min(), Y_predict.max()],[Y_predict.min(), Y_predict.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. True Values')
plt.show()
