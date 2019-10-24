import random
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np 

import hyperopt

a = np.random.rand(10,50)
print(str(a))