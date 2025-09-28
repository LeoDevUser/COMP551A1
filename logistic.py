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
from sklearn.model_selection import train_test_split

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
Y = breast_cancer_wisconsin_diagnostic[['Diagnosis']] #target

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

#get correlation matrix
corr = abs(X.corr())

#plot correlation matrix
plt.figure(figsize=(8,6))
sns.heatmap(corr, #plot heatmap for correlation matrix of mean variables for this dataset     
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5, 
            #xticklabels= ['Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', "Concavity", "Concave Points", "Symmetry", "Fractal Dimension" ], 
            #yticklabels= ['Diagnosis', 'Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', "Concavity", "Concave Points", "Symmetry", "Fractal Dimension" ]
            )
plt.show()

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

wdbc_bar = plt.bar(['Benign = 0', 'Malignant = 1'], Y.value_counts(), color=['tab:red', 'tab:blue']) #bar chart of diagnosis
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

#calculate what % of points are correctly categorized (eg. y_predicted>0.5 vs. y_predicted<0.5)
def logistic_success_rate(y_predicted, y):
    N = len(y_predicted)
    y = np.array(y)
    yp = y_predicted.round() #round to either 0 or 1 
    error = abs(yp-y) #if 0 then it was correctly predicted
    return round((N-error.sum())/N*100, 2)

def test_logistic_regression(x, y, testsplits): 
    train_pts = []
    test_pts = []
    for split in testsplits:
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split)
            model = LogisticRegression().fit(X_train, y_train)
            test = model.predict(X_test)
            performance = model.predict(X_train)               
            error_train = logistic_success_rate(performance, y_train)
            error_test =logistic_success_rate(test, y_test)
            train_pts.append([split, error_train])
            test_pts.append([split, error_test])
    print(train)
    train_error = []
    test_error = []
    for split in testsplits:
        train_avg = sum(train[split])/10
        train_SE = stats.sem(train[split])
        train_error.append([train_avg, train_SE])
        test_avg = sum(test[split])/10
        test_SE = stats.sem(test[split])
        test_error.append([test_avg, test_SE])
    return [train_error, test_error]

#train model
#Split Data
train_range = int(args.split / 100 * len(X)) #lenX is number of instances
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

train = len(X) - train_range
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

#this is Laurel working on the graph of performance by split, not done yet
splits = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.15, 0.1, 0.05]
#ex1 = test_logistic_regression(X, Y, splits)
#plt.errorbar(splits, ex1[0][0], yerr=ex1[0][1])
##plt.plot(splits, ex1[0][0], label= "PCT Correct of Training Data")
#plt.errorbar(splits, ex1[1][0], yerr=ex1[1][1])
#plt.plot(splits, ex1[1][0], label="PCT Correct of Testing Data")
#plt.legend(loc="upper right")
#plt.xlabel("Training Split")
#plt.ylabel("PCT Correctly Categorized")
#plt.show()
