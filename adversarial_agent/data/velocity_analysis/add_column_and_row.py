import pandas as pd
import os

def process_csv(filename, value=30):

    columns_names = ['Evx', 'Evy', 'Avx', 'Avy', 'episode', 'step']
    # Load the CSV file
    df = pd.read_csv(filename)
    
    episode = df['episode'].unique()[0]
    print(episode)

    # Create a new row with specified values
    new_row = pd.DataFrame({'Evx': [25], 'Evy': [0], 'Avx': [25], 'Avy': [0], 'episode': episode , 'step' : 0})
    df = pd.concat([new_row, df], ignore_index=True)
    
    last_step = df.iloc[-1]['step']
    #print('l: ', int(last_step))
    
    
    # Check if DataFrame has less than 30 rows
    if len(df) < 30:
        # Fill the rest of the rows with the values of the last row
        for i in range(len(df), 31):
            df.loc[i] = df.iloc[-1]
            df.loc[i]['step'] = last_step+1
            
            
            last_step+=1
    print(df)
            
    
    new_filename = os.path.splitext(filename)[0] + '_modified.csv'
    # Save the modified DataFrame back to CSV
    df.to_csv(new_filename, index=False)

# Define the folder path containing the CSV files
#folder_path = '/code/v_analysis/test_retrained_agent/run_20231006-153212_36081/episodes_retrained_agent'

#folder_path = 'code/v_analysis/test_agent_2_v2/run_20231006-153212_36081/episodes_test_agent_2'

folder_path = '/code/v_analysis/test_agent_1/run_20230914-170338_43658/episodes_agent_1'


# Loop through files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        process_csv(file_path)



