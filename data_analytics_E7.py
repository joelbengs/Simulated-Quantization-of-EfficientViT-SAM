import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from adjustText import adjust_text
from configs.quant_backbones_zoo import REGISTERED_BACKBONE_DESCRIPTIONS
from configs.quant_backbones_zoo import SIMPLE_REGISTERED_BACKBONE_DESCRIPTIONS


def read_pickle_to_dataframe(file_path, file_name) -> pd.DataFrame:
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]
        file_string = f'{file_path}/{file_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='From pickle file to plots')
    parser.add_argument('--file_path', type=str, default='results_storage', help='The directory of the pickle file')
    parser.add_argument('--file_name', type=str, help='The name of the pickle file to analyse')
    args = parser.parse_args()

    # load the data from storage
    df, _ = read_pickle_to_dataframe(args.file_path, args.file_name)

    pd.set_option('display.max_rows', None)
    print("data analytics script is working with the following data:")
    print(df.columns)

    # Print data overview
    # selected_columns = ['backbone_version', 'all'
    #                    'number of quantized params']
    #intersection_columns = [col for col in selected_columns if col in df.columns]
    #selected_data = df[intersection_columns]
    # Format 'number of quantized params' with thousands separator
    #if 'number of quantized params' in selected_data.columns:
    #    selected_data['number of quantized params'] = selected_data['number of quantized params'].apply(lambda x: '{:,}'.format(int(x)))


    #print(selected_data.to_string(index=False))

    ################################
    ###  Experiment 7 - simualted activation quantization       ###
    ################################

    ''' Function that plots blockwise and layerwise benchmarking '''
    def plot(df, title: str, xlabel: str, name: str, performance_measure: str = 'all', layerwise=False, rotate=False, zoom=False):
        baseline_value=78.509
        zoom_lower_limit=77
        zoom_upper_limit=79
        texts = []

        fig, ax = plt.subplots(figsize=(25,5))
        if zoom:
            # Create a color array that holds the color for each dot
            colors = df[performance_measure].apply(lambda y: 'red' if y < zoom_lower_limit else 'blue')
            # Create a new column 'adjusted_all' that holds the adjusted y-values
            df['adjusted_performance_measure'] = df[performance_measure].apply(lambda y: zoom_lower_limit + 0.1 if y < zoom_lower_limit else y)
            # Use scatter to draw dots with the adjusted y-values and colors
            ax.scatter(df['backbone_version'], df['adjusted_performance_measure'], c=colors)
        else:
            ax.plot(df['backbone_version'], df[performance_measure])


        if layerwise:
            # Add description to each data point where 'all' score is substantially lower than baseline
            for i, val in df.iterrows(): # index and row data
                if zoom:
                    if val[performance_measure] < baseline_value-0.1:
                        # using adjustText library to avoid overlapping text
                        texts.append(ax.text(val['backbone_version'], val['adjusted_performance_measure'], f"{SIMPLE_REGISTERED_BACKBONE_DESCRIPTIONS[val['backbone_version']]}"))
                else:
                    if val[performance_measure] < baseline_value-0.5:
                        ax.text(val['backbone_version'], val[performance_measure] - 1, f"{SIMPLE_REGISTERED_BACKBONE_DESCRIPTIONS[val['backbone_version']]}")
            if zoom:
                adjust_text(texts)  # adjust text to minimize overlaps

            
        # vertical lines at each x mark
        for x in df['backbone_version']:
            ax.axvline(x, color='lightgray', linestyle='dotted')
        ax.axhline(baseline_value, color='red', linestyle='dotted')
        ax.text(0, baseline_value - 0.1 if zoom else baseline_value - 1, 'Baseline', va='top', ha="right")


        ax.set_xlabel(xlabel)
        ax.set_ylabel('Ground truth box prompt "all" score (higher is better)')
        if rotate:
            plt.xticks(rotation=70)
        if zoom:
            ax.set_ylim([zoom_lower_limit,zoom_upper_limit])

        plt.suptitle(title)
        plt.savefig(f'./plots/graphs/{name}.png', bbox_inches='tight')
        plt.close()


    df_block = df.copy()
    df_layer = df.copy()

    df_block['model'] = df_block['model'].str.replace('_quant', '')
    df_block['model'] = df_block['model'].str.upper()
    df_block['backbone_version'] = df_block['backbone_version'].str.replace('L0:', '')
    df_block['backbone_version'] = df_block['backbone_version'].str.replace('stage', 'stage ')
    df_block['backbone_version'] = df_block['backbone_version'].str.replace(':', '\nblock ', 1) # only first occurance
    df_block['backbone_version'] = df_block['backbone_version'].str.replace(':', '\nlayer ')

    df_layer['model'] = df_layer['model'].str.replace('_quant', '')
    df_layer['model'] = df_layer['model'].str.upper()
    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace('L0:', '')
    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace('stage', '')

    # split data: stage:block:x vs stage:block:layer
    df_block = df_block[df_block['backbone_version'].str.endswith('x')] # remove layerwise experiments
    df_layer = df_layer[~df_layer['backbone_version'].str.endswith('x')] # remove blockwise experiments

    plot(df_block,
           title='Block-wise analysis of weight+activation quantization to INT8 of EfficientViT-SAM L0 image encoder',
            xlabel='L0 base model with only one block quantized. All layers of the block are quantized',
            name=f'{args.file_name}_block',
            performance_measure = "box_all",
            )
    
    plot(df_layer,
           title='Layer-wise analysis of weight+activation quantization to INT8 of EfficientViT-SAM L0 image encoder',
           xlabel='L0 base model with only one layer quantized. Naming scheme is stage:block:layer',
           name=f'{args.file_name}_layer',
           performance_measure = "box_all",
           layerwise=True,
           rotate=True,
           )
    
    plot(df_block,
           title='Block-wise analysis of weight+activation quantization to INT8 of EfficientViT-SAM L0 image encoder',
            xlabel='L0 base model with only one block quantized. All layers of the block are quantized',
            name=f'{args.file_name}_block_zoom',
            performance_measure = "box_all",
            zoom=True,
            )
    
    plot(df_layer,
           title='Layer-wise analysis of weight+activation quantization to INT8 of EfficientViT-SAM L0 image encoder',
           xlabel='L0 base model with only one layer quantized. Naming scheme is stage:block:layer',
           name=f'{args.file_name}_layer_zoom',
           performance_measure = "box_all",
           layerwise=True,
           rotate=True,
           zoom=True,
           )


