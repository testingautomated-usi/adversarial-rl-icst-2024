import pandas as pd
import os

def concatenate_and_calculate_average(folder_path):
    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate all DataFrames into one
    combined_df = pd.concat(dfs, ignore_index=True)

    # Calculate the average for each unique value in the 'step' column
    average_by_step = combined_df.groupby('step').mean()
    std_by_step = combined_df.groupby('step').std()
    
    evx_values_step_7 = combined_df[combined_df['step'] == 5]['Evx']
    print('test_i')
    print(list(evx_values_step_7.values))
    
    #steps = average_by_step['step'].values
    mean_e_vx = average_by_step['Evx'].values
    mean_a_vx = average_by_step['Avx'].values
    std_e_vx = std_by_step['Evx'].values
    std_a_vx = std_by_step['Avx'].values
   
    steps = range(len(mean_e_vx))
    dic = {'steps' : steps, 'mean Evx' : mean_e_vx, 'mean Avx' : mean_a_vx,
    'std Evx' : std_e_vx, 'std Avx': std_a_vx}
    
    df_new = pd.DataFrame (dic)
    
    return df_new
    
    
    
    return average_by_step

# Define the folder path containing the CSV files
#folder_path = 'code/v_analysis/test_retrained_agent/run_20231006-153212_36081/episodes_retrained_agent/modified'  # Replace with your actual folder path

#folder_path = 'code/v_analysis/test_agent_2_v2/run_20231006-153212_36081/episodes_test_agent_2/modified'

folder_path = 'code/v_analysis/test_agent_1/run_20230914-170338_43658/episodes_agent_1/modified'

result_df = concatenate_and_calculate_average(folder_path)

#result_df.to_csv('./retrained_agent_new.csv')
#result_df.to_csv('./test_agent_2_new.csv')

result_df.to_csv('./test_agent_1.csv')
# Print or use the 'result_df' as needed
#print(result_df)

