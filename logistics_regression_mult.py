

import numpy as np

class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        m = X.shape[0]

        for _ in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - self._sigmoid(output)
            self.w += self.eta / m * errors.dot(X)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

X = np.array([[-2, 2],[-3, 0],[2, -1],[1, -4]])
y = np.array([1,1,0,0])
logi = LogisticRegression().fit(X, y)
print(logi.predict(X))

class LogisticRegressionOVR(object):
	def __init__(self, eta=0.1, n_iter=50):
	    self.eta = eta
	    self.n_iter = n_iter

	def fit(self, X, y):
		X = np.insert(X, 0, 1, axis=1)
		self.w = []
		m = X.shape[0]

		for i in np.unique(y):
			y_copy = np.where(y == i, 1, 0)
			w = np.ones(X.shape[1])
			for _ in range(self.n_iter):
				output = X.dot(w)
				errors = y_copy - self._sigmoid(output)
				w += self.eta / m * errors.dot(X)
			self.w.append((w, i))
		return self
	def _predict_one(self, x):
		return max((x.dot(w), c) for w, c in self.w)[1]

	def predict(self,X):
		return [self._predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

	def _sigmoid(self,x):
		return 1/(1+np.exp(-x))
	def score(self,X,y):
		print ("Prediction: ",self.predict(X))
		print ("y value: ",y)
		return sum(self.predict(X)==y)/len(y)

from sklearn import datasets
np.set_printoptions(precision=3)
iris = datasets.load_iris()
X = iris.data
y = iris.target
logi = LogisticRegressionOVR(n_iter=1000).fit(X, y)
print(logi.w)



from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp = \
    train_test_split(iris.data, iris.target, test_size=.4)
X_validation, X_test, y_validation, y_test = \
    train_test_split(X_temp, y_temp, test_size=.5)

logi = LogisticRegressionOVR(n_iter=1000).fit(X_train, y_train)

print(logi.score(X_train, y_train))
print(logi.score(X_validation, y_validation))



