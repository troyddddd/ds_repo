

import numpy as np
y = [1,2,3]
for i in np.unique(y):
	print (i)
	y_copy = np.where(y==i,1,0)
	print (y_copy)
