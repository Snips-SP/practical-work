from backend import server
from backend.ml_model.experiments import testing_generation, testing_conversion

# Start webserver
# server.run()


testing_generation()
testing_conversion()