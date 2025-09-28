import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.special import expit as logistic
import argparse
import seaborn as sns
matplotlib.use('Qt5Agg') #Set GUI backend for plots
from scipy import stats

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
parser = argparse.ArgumentParser(usage='sgdlinear.py [--split|--batch|--alpha] [values]')
#add args
parser.add_argument('--split', type=check_int, help='represents the fraction used as the training set', default=80)
parser.add_argument('--alpha', type=int, help='The learning rate', default=10)
parser.add_argument('--batch', type=check_int, help='Represents the fraction of the training set that is to be batched', default=50)
args = parser.parse_args()
if args.split == 100:
    args.split = 99
if args.alpha == 0:
    args.alpha = 1
args.alpha = abs(args.alpha)

#set headers
wdbcHeaders = ["ID", "Diagnosis"]
features = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
for i in range(len(features)):
    wdbcHeaders.insert(i+2, features[i]+"1") #average value of the feature
    wdbcHeaders.insert(2*i+3, features[i]+"2") #standard error of the feature
    wdbcHeaders.insert(2*i+4, features[i]+"3") #worst/largest of the feature

# fetch dataset
#breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
breast_cancer_wisconsin_diagnostic = pd.read_csv('wdbc_diagnosis.csv', header=None, names=wdbcHeaders)
pd.set_option('display.max_columns', None)

# data (as pandas dataframes)
#X = breast_cancer_wisconsin_diagnostic.data.features
#Y = breast_cancer_wisconsin_diagnostic.data.targets.copy()
X = breast_cancer_wisconsin_diagnostic.drop(columns =['Diagnosis', 'ID']) #features
X = X.drop(X.iloc[:, 10:20], axis=1) #drop standard error features
Y = breast_cancer_wisconsin_diagnostic[['Diagnosis']].copy() #target

# Convert string labels to numeric: 'M' (malignant) = 1, 'B' (benign) = 0
Y['Diagnosis'] = Y['Diagnosis'].map({'M': 1, 'B': 0})

#get correlation matrix
corr = abs(X.corr())
#get upper triangular matrix
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
#find features with correlation greater than 0.9 with each other 
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
#drop highly correlated features
X = X.drop(to_drop, axis=1)

#drop outliers with z score greater than 3 
z = np.abs(stats.zscore(X))
#~0.2% (0.1 above and 0.1 below the mean) of data in a normally distributed dataset will fall in this range
outlier_indices = np.where(z > 3)[0]
X = X.drop(outlier_indices)
Y = Y.drop(outlier_indices)

wdbc_bar = plt.bar(['Benign = 0', 'Malignant = 1'], Y.value_counts(), color=['tab:red', 'tab:blue']) #bar chart of diagnosis
plt.title('Malignant vs Benign Tumors in the Dataset After Preprocessing')
plt.xlabel('Diagnosis')
plt.bar_label(wdbc_bar, labels=Y.value_counts(), label_type='center')
plt.ylabel('Count')
plt.show()

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
        t = 1
        global batch_size
        batch_size = int(args.batch * N / 100)
        #gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iter:
            random_indices = np.random.choice(N,size=batch_size,replace=False)
            X_batch = x[random_indices]
            Y_batch = y[random_indices]
            g = self.gradient(X_batch,Y_batch)
            self.w = self.w - self.learning_rate /t * g
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

#calculate what % of points are correctly categorized (eg. y_predicted>0.5 vs. y_predicted<0.5)
def logistic_success_rate(y_predicted, y):
    N = len(y_predicted)
    y = np.array(y)
    yp = y_predicted.round() #round to either 0 or 1 
    error = abs(yp-y) #if 0 then it was correctly predicted
    return round((N-error.sum())/N*100, 2)

def test_logistic_regression(x, y, split_percent, learning_rate): 
    train_range = int(len(x) * split_percent / 100)
    
    # Split data manually
    X_train = x.iloc[:train_range]
    X_test = x.iloc[train_range:]
    y_train = y.iloc[:train_range]
    y_test = y.iloc[train_range:]
    
    model = LogisticRegression(learning_rate=learning_rate).fit(X_train, y_train)
    yh = model.predict(X_test)
    y = y_test['Diagnosis'].to_numpy()

    matching = (yh.round() == y).sum() #Number of correct classifications
    test_total = len(y_test)
    
    # Print results (only test data)
    print(f'Results for a {args.split}/{100-args.split} train/test split and batch size {batch_size} with learning rate {args.alpha}:')
    print(f'{matching}/{test_total} correct classifications ({round(matching * 100 / test_total, 2)}% accuracy)')
    
    return matching, test_total

#run the test using command line arguments
test_logistic_regression(X, Y, args.split, args.alpha)
