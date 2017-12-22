README
==============================================================================================
Mitch Negus
3032146443
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
CODE - HW01 - CompSci 289A

The code for this assignment includes 3 Jupyter notebooks (CS289A_HW01_MNIST.ipynb, CS289A_HW01_spam.ipynb, CS289_HW01_CIFAR-10.ipynb) that include code for each dataset, as well as 2 supplemental python modules (HW01_utils.py, trainfunctions.py). The description of each file is as follows:

* CS289A_HW01_MNIST.ipynb		-main code for partioning, training, & testing MNIST data
* CS289A_HW01_spam.ipynb		-main code for partioning, training, & testing spam data
* CS289_HW01_CIFAR-10.ipynb	-main code for partioning, training, & testing CIFAR data
* HW01_utils.py			-utility module with functions to load & format data as arrays
* trainfunctions.py		-module with functions for training and testing with SVM

*both .py modules must be in the PYTHONPATH for the Jupyter notebooks to work.

The Jupyter notebooks take in 3 input parameters: an integer validation set size (valsetsize), an array with numbers of training samples to use (samples), and an array of hyperparameters (hyperparams). As of HW submission, the hyperparameters array can only take a list of C-values for the SVC algorithm.

In addition to these 3 inputs, 3+ paths must be specified:   
(1)_LOCAL_PATH, the path to the directory containing all information, code, etc. pertaining to this homework  
(2)_DATA_PATH, the path (starting from _LOCAL_PATH) to the directory containing all training and test data  
(3)	datafilepath (in spam only) - the path to spam training/test data file  
	trainpath (in MNIST and CIFAR only) - the path to either MNIST or CIFAR training data file  
	testpath (in MNIST only) - the path to MNIST test data file (for Kaggle)  
(Ex: the absolute path to the MNIST training data should be given by _LOCAL_PATH\_DATAPATH\trainpath)  

Outputs of the code depend on which cells are executed in the Jupyter notebook. 
1) Necessary for successful generation of all future outputs is an array of accuracies for each training sample count and C-value (the array has shape [len(samples) x len(hyperparams)]). The cell which produces this output will also print accuracies to std. output or the console.
2) For MNIST and spam data, the Jupyter notebook can generate a csv file containing predictions for a input data set based on the complete set of training data provided (including data that had previously been separated for validation).
3) All three notebooks can generate plots of error (error = 1-accuracy) as a function of training sample size.

*To cover a wide range of C-values in hyperparameter testing, a set of 20 logarithmically evenly-spaced C-values were chosen. This decision resulted in data where an accuracy for the default C=1 was not calculated. Without an accuracy for C=1, the error plots feature the error for each sample-count for the closest value of C to 1 in the set of 20 values.
