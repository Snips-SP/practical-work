# from backend import server
# from backend.ml_model.experiments import testing_generation, training_test, testing_generation_function, testing_conversion

import os
from backend.ml_model.train import train

# Start webserver
# server.run()

# training_test()

train(continue_from=os.path.join('backend', 'ml_model', 'runs', 'GPT2_Model_20'))



