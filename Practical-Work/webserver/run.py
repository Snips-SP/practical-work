
# from backend import server
from backend.ml_model.experiments import testing_generation, training_test
# from backend.ml_model.encode import encode_dataset

import os
from backend.ml_model.train import train_simple

# Start webserver
# server.run()

default_path = os.path.join('backend', 'ml_model')
train_simple(default_path) #, os.path.join('backend', 'ml_model', 'runs', 'GPT2_Model_7'))

#training_test()

# testing_generation()

# import os

# if __name__ == '__main__':
#    os.chdir(os.path.join('backend', 'ml_model'))
#
#    encode_dataset('ldp_5_dataset', da=1)


