import argparse
import pandas as pd
import pickle
import os

def read_pickle_to_dataframe(file_path, script_name) -> pd.DataFrame:
        script_name = os.path.basename(script_name)
        script_name = os.path.splitext(script_name)[0]
        file_string = f'{file_path}/{script_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read pickle file and convert it to pandas dataframe')
    parser.add_argument('--pickle_file_path', type=str, default='results', help='The directory of the pickle file')
    parser.add_argument('--script_name', type=str, help='The name of the script')

    args = parser.parse_args()

    df, file_string = read_pickle_to_dataframe(args.pickle_file_path, args.script_name)

    # Assuming you have a list of column names
    selected_columns = ['model',
                        'prompt_type',
                        'quantize_W',
                        'observer_method_W',
                        'quantize_method_W',
                        'all', 'large', 'medium', 'small']

    # Filter the selected_columns list to only include column names that exist in the DataFrame
    intersection_columns = [col for col in selected_columns if col in df.columns]

    # Select these columns from the dataframe
    selected_data = df[intersection_columns]

    # Print the selected data
    print(selected_data)

    #print(f"\nThe head of the dataframe {file_string} is: ")
    #print(df.head())
    #print(f"\nThe tail of the dataframe {file_string} is: ")
    #print(df.tail())