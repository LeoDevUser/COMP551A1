import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.special import expit as logistic
import argparse
matplotlib.use('Qt5Agg') #Set GUI backend for plots

#parse arguments
batch_size = 0

def check_int(value):
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")

    if ivalue <= 0 or ivalue > 100:
        raise argparse.ArgumentTypeError(f"'{value}' must be an integer between 0 and 100 (exclusive).")
    
    return ivalue
#create parser
parser = argparse.ArgumentParser(usage='logistic.py [--split|--batch] [values]')
#add args
parser.add_argument('--split', type=check_int, help='represents the fraction used as the training set', default=80)
parser.add_argument('--alpha', type=int, help='The learning rate', default=0.5)
args = parser.parse_args()
if args.split == 100:
    args.split = 99
if args.alpha == 0:
    args.alpha = 1
args.alpha = abs(args.alpha)

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
pd.set_option('display.max_columns', None)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
Y = breast_cancer_wisconsin_diagnostic.data.targets.copy()

# metadata
#print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
#print(breast_cancer_wisconsin_diagnostic.variables)


# variable information
#print(X[X.eq('?').any(axis=1)])
#print(X.info())
#print(X.describe())
#print(Y.info())
#print(Y.describe())

#Turn M and B into binary values
diagnosis_map = {'M' : 1, 'B' : 0}
Y['Diagnosis'] = Y['Diagnosis'].map(diagnosis_map)

#logistic = lambda z : 1. / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, add_bias=True, learning_rate=1., epsilon=1e-4, max_iter=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self,x,y):
        #Convert to numpy arrays if necessary
        if hasattr(x, 'values'):
            x = x.values
        if hasattr(y, 'values'):
            y = y.values.flatten()
        if x.ndim == 1:
            x = x[:,None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        g = np.inf
        t = 0
        #gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iter:
            g = self.gradient(x,y)
            self.w = self.w - self.learning_rate * g
            t += 1

        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self

    def predict(self,x):
        if hasattr(x, 'values'):
            x = x.values
        if x.ndim == 1:
            x = x[:,None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))
        return yh

    def gradient(self,x,y):
        N = x.shape[0]
        yh = logistic(np.dot(x,self.w))
        grad = np.dot(x.T, yh - y) / N
        return grad

#train model
#Split Data
train_range = int(args.split / 100 * 569) #569 is the number of instances
X_train = X.iloc[:train_range]
X_predict = X.iloc[train_range:]
Y_train = Y.iloc[:train_range]
Y_predict = Y.iloc[train_range:]

#Scale Features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_predict = scaler.transform(X_predict)

#Fit and Predict
model = LogisticRegression(learning_rate=args.alpha)
yh = model.fit(X_train,Y_train).predict(X_predict)

train = 569 - train_range
y = Y_predict['Diagnosis'].to_numpy()
matching = (yh == y).sum() #Number of correct classifications
print(f'Results for a {args.split}/{100-args.split} train/test split:')
print(f'{matching}/{train} correct classifications ({round(matching * 100 / train,2)}% accuracy)')
x_plot = np.linspace(0,train-1, train)
plt.plot(x_plot,yh,'r.')
plt.plot(x_plot, Y_predict, '.')
plt.xlabel('Instance')
plt.ylabel('Classification')
plt.title('Logistic Regression Breast Cancer Classification')
plt.show()
