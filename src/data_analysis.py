import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# File path of the .mat file
file_path = '/Users/shreyas/Desktop/UoU/Independent-Study/Airflow-rate-prediction/dataset_new/FanPower_1.6V/temp_2025-3-7-19-14-21_21.4_35_13.6_.mat'

# Load the .mat file
mat_data = scipy.io.loadmat(file_path)

# Check the keys in the loaded dictionary to get an idea of the data structure
print("Keys in the MAT file:", mat_data.keys())

# Filter out the metadata keys (__header__, __version__, __globals__)
data_keys = [key for key in mat_data.keys() if not key.startswith('__')]

# Inspect each data key to see the structure of the actual data
for key in data_keys:
    print(f"\nKey: {key}")
    print(f"Type: {type(mat_data[key])}")
    print(f"Shape/Size: {np.shape(mat_data[key])}")
    print(f"Data type: {mat_data[key].dtype if hasattr(mat_data[key], 'dtype') else 'N/A'}")
    print("-" * 40)

# Now let's extract the 'TempFrames' variable (or whatever the actual data key is)
variable_name = 'TempFrames'

if variable_name in mat_data:
    temp_frames_data = mat_data[variable_name]
    
    # Show the basic info about this variable
    print(f"\nDetails for variable: {variable_name}")
    print(f"Type: {type(temp_frames_data)}")
    print(f"Shape: {temp_frames_data.shape}")
    print(f"Data type: {temp_frames_data.dtype}")
    
    # If it's a numpy array, let's explore it further
    if isinstance(temp_frames_data, np.ndarray):
        print(f"Number of dimensions: {temp_frames_data.ndim}")
        print(f"Size of the data: {temp_frames_data.size}")
        print(f"Min value: {np.min(temp_frames_data)}")
        print(f"Max value: {np.max(temp_frames_data)}")
        print(f"Mean value: {np.mean(temp_frames_data)}")
        print(f"Standard deviation: {np.std(temp_frames_data)}")
    
    # Visualizing the data (if it is 2D or 3D)
    if temp_frames_data.ndim == 2:  # For 2D data, plot a simple line or matrix plot
        plt.plot(temp_frames_data)
        plt.title('TempFrames Data Analysis')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.show()
    elif temp_frames_data.ndim == 3:  # For 3D data (e.g., time series or frames)
        plt.imshow(temp_frames_data[0], cmap='hot', interpolation='nearest')  # Example: first frame
        plt.colorbar()
        plt.title('First Frame of TempFrames Data')
        plt.show()

else:
    print(f"Variable '{variable_name}' not found in the MAT file. Please check the variable name.")
