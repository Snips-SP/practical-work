# from backend import server
# from backend.ml_model.experiments import dataloader_test

import os
from backend.ml_model.train import train

# Start webserver
# server.run()

train(os.path.join('backend', 'ml_model'), os.path.join('backend', 'ml_model', 'runs', 'GPT2_Model_7'))



