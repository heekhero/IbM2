import numpy as np
import torch
import pickle

import numpy as np


a = ((2,1,2), (2,4,-1), (2, -1, -5))

c = max(a, key=lambda x:(x[0], -x[2]))
print(c)