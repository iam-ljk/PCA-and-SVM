import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn import preprocessing
import math


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	# load the data
	train_df = pd.read_csv('data/mnist_train.csv')
	test_df = pd.read_csv('data/mnist_test.csv')

	X_train = train_df.drop('label', axis=1).values
	y_train = train_df['label'].values

	X_test = test_df.drop('label', axis=1).values
	y_test = test_df['label'].values

	return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
	# normalize the data
	# print(preprocessing.normalize(X_train))
	# print(X_train)
	for j in range(0, len(X_train[0])):
	# for j in range(0, 50):
		_min = 256
		_max = 0
		for i in range(0, len(X_train)) :
		# for i in range(0, 1000) :
			_min = min(_min, X_train[i][j])
			_max = max(_max, X_train[i][j])
		for i in range(0,len(X_train)) :
		# for i in range(0,1000) :		
			diff = _max - _min
			if diff != 0 :
				X_train[i][j] = (X_train[i][j] - _min) / diff
			else :
				X_train[i][j] = (X_train[i][j] - _min)
			X_train[i][j] = (2.0*X_train[i][j]) - 1.0
	# print(X_train[0][0])
	# for j in range(0,len(X_train[0])) :
	# 	for i in range(0,len(X_train)) :
	# 		if X_train[i][j] < 1 and X_train[i][j] > -1 and X_train[i][j] != 0 :
	# 			print(i,j)
	# 			break
	print(X_train[12352][12], X_train[12352][13])
	


	return X_train, X_test


def plot_metrics(metrics) -> None:
    # plot and save the results
    raise NotImplementedError
