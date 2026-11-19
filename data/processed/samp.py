import numpy as np

keys_to_extract = {'X_tr', 'y_tr', 'X_val', 'y_val', 'X_test', 'y_test', 'features'}
extracted_data = {}

with np.load('fd001_windows.npz') as data:
    # Safely pull only the keys that actually exist in the file
    for key in keys_to_extract:
        if key in data.files:
            extracted_data[key] = data[key]

# Access your arrays from the new dictionary
print(extracted_data['y_tr'])