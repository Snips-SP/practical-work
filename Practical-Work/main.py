import numpy as np

# Open the .npz file
file_path = "ldp_5_dataset/01.npz"
data = np.load(file_path)

# Access the arrays
for array_name in data:
    print(f"Array name: {array_name}, Data: {data[array_name]}")