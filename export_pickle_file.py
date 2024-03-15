import argparse
import pandas as pd
import pickle
import os

def export_pickle_to_excel(file_path, script_name) -> None:
    script_name = os.path.basename(script_name)
    script_name = os.path.splitext(script_name)[0]
    file_string = f'{file_path}/{script_name}.pkl'
    excel_file_string = f'{file_path}/{script_name}.xlsx'

    with open(file_string, 'rb') as file: # read binary
        data = pickle.load(file)
    df = pd.DataFrame(data)
    df.to_excel(excel_file_string, sheet_name=script_name)
    print(f"Exported pickle file to excel file named {script_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read pickle file and convert it to pandas dataframe')
    parser.add_argument('--pickle_file_path', type=str, default='results', help='The directory of the pickle file')
    parser.add_argument('--script_name', type=str, help='The name of the script')

    args = parser.parse_args()
    
    export_pickle_to_excel(args.pickle_file_path, args.script_name)

