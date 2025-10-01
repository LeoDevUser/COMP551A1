First do pip install -r requirements.txt

Now we can run any of the 4 models linear.py sgdlinear.py logistic.py sgdlinear.py
the sgd prefix indicates that is the stochastic gradient descent version of the model

you can do python [model].py -h to see the accepted command-line arguments for each model

For example:
$ python sgdlogistic.py -h
usage: sgdlinear.py [--split|--batch|--alpha] [values]

options:
  -h, --help     show this help message and exit
  --split SPLIT  represents the fraction used as the training set
  --alpha ALPHA  The learning rate
  --batch BATCH  Represents the fraction of the training set that is to be batched

for the .ipynb to run, they must be able to access the .csv files
