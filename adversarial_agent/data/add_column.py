import os
import argparse
import pandas as pd


#folder_path = '/code/data/test_agent_2_select'

#files = os.listdir(folder_path)

#column_data = 'agent_2'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Add an extra column into the files")
    parser.add_argument("file_path", type=str, help="Path to the CSV file.")
    parser.add_argument("column", type=str, help="name of the column to be included")
    #parser.add_argument('save_path', type=str, help="Path to save the new CSV file.")

    args = parser.parse_args()
    folder_path = args.file_path
    column_data = args.column

    for dirpath, dirnames, filenames in os.walk(folder_path):

        for file in filenames:

            if file.endswith('.csv') and 'crashed' in file and 'selected' not in file:
            #if file.endswith('.csv') and 'crashed_selected' in file:
                print(file)
                
                # Construct the full file path
                full_path = os.path.join(dirpath, file)

                print(full_path)
                    
                # Load the existing CSV file
                df = pd.read_csv(full_path)

                # Add a new column 'type' with value 'random'
                df['type'] = column_data
                
                csv_file_name = os.path.splitext(file)[0]
                new_file_name = csv_file_name + '_typed_not_selected' + '.csv'
                new_save_folder = os.path.join(dirpath, 'typed_not_selected')
                print('nsf: ', new_save_folder)
                os.makedirs(new_save_folder, exist_ok=True)

                new_file_name = os.path.join(new_save_folder,new_file_name)
                print('nfn: ', new_file_name)
                
                # Save the updated DataFrame to a new CSV file
                #output_file = os.path.join(folder_path, f'typed_{file}')
                df.to_csv(new_file_name, index=False)

                print(f'New CSV file saved as {new_file_name}')
                




