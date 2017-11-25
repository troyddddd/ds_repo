

# -*- coding: utf-8 -*-
#rnn from scratch by Yijun D.

import numpy as np

data = open('kafka.txt','r').read()

chars = list(set(data))
data_size,vocab_size = len(data), len(chars)
print ('data has %d chars, %d unique' % (data_size,vocab_size))

