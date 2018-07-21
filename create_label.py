'''Creates a pickle file with labels derived from all the sub-directory names'''

import os
import pickle

with open('/home/aravind/Assesment/labels.pickle', 'wb') as fp:
    pickle.dump(sorted(os.listdir('/home/aravind/Assesment/Data/test')), fp)
