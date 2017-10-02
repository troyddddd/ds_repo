



#my own practice with logistics regression binary classification in py


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

num_obs = 5000

x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_obs)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_obs)

simulated_separab_features = np.vstack((x1,x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_obs),np.ones(num_obs)))

plt.figure(figsize = (12,8))
plt.scatter(simulated_separab_features[:,0],simulated_separab_features[:,1],c = simulated_labels,alpha = .4)


def sigmoid(x):
	return 1/(1+np.exp(-x))

def log_like(f,t,w):
	x = np.dot(f,w)
	ll = np.sum(t*x - np.log(1+np.exp(x)))
	return ll

def logistics_regression(f,t,num_steps,learning_rate,add_intercept = False):
	if add_intercept:
		intercept = np.ones((f.shape[0],1))
		f = np.hstack((intercept,f))
	weights = np.zeros(f.shape[1])

	for step in xrange(num_steps):
		scores = np.dot(f,weights)
		predictions = sigmoid(scores)

		output_error = t - predictions
		gradient = np.dot(f.T,output_error)
		weights += learning_rate*gradient

		if step%10000 == 0:
			print log_like(f,t,weights)
	return weights

weights = logistics_regression(simulated_separab_features,simulated_labels,num_steps = 300000,learning_rate = 5e-5,add_intercept = True)
data_with_intercept = np.hstack((np.ones((simulated_separab_features.shape[0], 1)),simulated_separab_features))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print 'Accuracy from scratch: {0}'.format((preds == simulated_labels).sum().astype(float) / len(preds))

plt.figure(figsize = (12, 8))
plt.scatter(simulated_separab_features[:, 0], simulated_separab_features[:, 1],c = preds == simulated_labels - 1, alpha = .8, s = 50)


plt.show()







