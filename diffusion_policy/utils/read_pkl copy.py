import zipfile
import pickle
import os
'''观察多个pkl'''
# Define the extraction directory
extract_dir = "/mnt/data-3/users/mengxinpan/code/diffusion_policy/data/real/bc_data_il/gello/3/"

# Loop through files 00055.pkl to 00065.pkl
for i in range(1, 30):  # 55 to 65 inclusive
    # Construct the file path with leading zeros (e.g., 00055.pkl)
    pkl_filename = f"{i:05d}.pkl"  # Ensures 5-digit format like 00055
    pkl_path = os.path.join(extract_dir, pkl_filename)

    # Check if the file exists
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found!")
    else:
        # Open and load the .pkl file
        with open(pkl_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        
        # Extract and print 'action' and 'joint_positions' if they exist in the data
        action = data.get('action', 'N/A')  # Use get() to avoid KeyError if key is missing
        joint_positions = data.get('joint_positions', 'N/A')
        print(f"\nFile: {pkl_filename}")
        # print("Action:", action)
        print("Joint Positions:", joint_positions)