import pandas as pd
import os

def split_csv_by_episode(input_file):
    # Load the CSV file
    columns_names = ['Evx', 'Evy', 'Avx', 'Avy', 'episode', 'step']
    df = pd.read_csv(input_file, names=columns_names )

    # Get unique values in the 'episode' column
    unique_episodes = df['episode'].unique()

    for episode in unique_episodes:
        # Create a new DataFrame containing only rows with the current episode value
        episode_df = df[df['episode'] == episode]
        input_file = os.path.splitext(input_file)[0]
        # Define the output filename
        output_filename = f'{input_file}_episode_{episode}.csv'
        #print(output_filename)
        # Save the DataFrame to a new CSV file
        episode_df.to_csv(output_filename, index=False)

def process_folder(folder_path):
    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        print(filename)
        if filename.endswith('.csv') and 'velocities' in filename:
            print(os.path.splitext(filename)[0])
            file_path = os.path.join(folder_path, filename)
            print('yes')
            split_csv_by_episode(file_path)

#folder_path = 'code/v_analysis/test_retrained_agent/run_20231006-153212_36081'  # Replace with your actual folder path
#folder_path = 'code/v_analysis/test_agent_1/run_20230914-170338_43658'

folder_path = 'code/v_analysis/test_agent_1/run_20230914-170338_43658'
process_folder(folder_path)
