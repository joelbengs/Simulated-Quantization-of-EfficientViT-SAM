import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from adjustText import adjust_text
from configs.quant_backbones_zoo import REGISTERED_BACKBONE_DESCRIPTIONS_LARGE, REGISTERED_BACKBONE_DESCRIPTIONS_XL


def read_pickle_to_dataframe(file_path, file_name) -> pd.DataFrame:
        file_name = os.path.basename(file_name)
        file_name = os.path.splitext(file_name)[0]
        file_string = f'{file_path}/{file_name}.pkl'

        with open(file_string, 'rb') as file: # read binary
            data = pickle.load(file)
        df = pd.DataFrame(data)
        return df, file_string

''' Function that plots blockwise and layerwise benchmarking '''
def plot(df, title: str, xlabel: str, name: str, model: str = 'L0', prompt_type: str = 'box', add_descriptions=True, rotate=False, zoom=False):
    baselines_box = { # measured empirically
                'L0':78.509,
                'L1':78.835,
                'L2':79.130,
                'XL0':79.752,
                'XL1':79.930,
            }
    
    if model.startswith('X'):
        description_dict = REGISTERED_BACKBONE_DESCRIPTIONS_XL
    else:
        description_dict = REGISTERED_BACKBONE_DESCRIPTIONS_LARGE

    baseline_value=baselines_box[model]
    zoom_lower_limit=77
    zoom_upper_limit= 80 # baseline_value + 0.2
    texts = []
    
    if prompt_type == 'box' or prompt_type == 'point':
        performance_measure = 'all'
    elif performance_measure == 'point_and_box':
        raise KeyError("plot function that can plot both point and box in one call not yet implemented")
    
    fig, ax = plt.subplots(figsize=(25,5))
    if zoom:
        # Create a color array that holds the color for each dot
        colors = df[performance_measure].apply(lambda y: 'red' if y < zoom_lower_limit else 'blue')
        # Create a new column 'adjusted_performance_measure' that holds the adjusted y-values
        df['adjusted_performance_measure'] = df[performance_measure].apply(lambda y: zoom_lower_limit + 0.1 if y < zoom_lower_limit else y)
        # Use scatter to draw dots with the adjusted y-values and colors
        ax.scatter(df['backbone_version'], df['adjusted_performance_measure'], c=colors)
    else:
        ax.plot(df['backbone_version'], df[performance_measure])


    if add_descriptions:
        # Add description to each data point where 'all' score is substantially lower than baseline
        for i, val in df.iterrows(): # index and row data
            if zoom:
                if val[performance_measure] < baseline_value-0.1:
                    # using adjustText library to avoid overlapping text
                    texts.append(ax.text(val['backbone_version'], val['adjusted_performance_measure'], f"{description_dict[val['backbone_version']]}"))
            else:
                if val[performance_measure] < baseline_value-1:
                    #ax.text(val['backbone_version'], val[performance_measure], f"{description_dict[val['backbone_version']]}")
                    texts.append(ax.text(val['backbone_version'], val[performance_measure], f"{description_dict[val['backbone_version']]}"))
        adjust_text(texts)  # adjust text to minimize overlaps

        
    # vertical lines at each x mark
    for x in df['backbone_version']:
        ax.axvline(x, color='lightgray', linestyle='dotted')
    ax.axhline(baseline_value, color='red', linestyle='dotted')
    ax.text(0, baseline_value - 0.1 if zoom else baseline_value - 1, 'Baseline', va='top', ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel('mIoU (higher is better)')
    if rotate:
        plt.xticks(rotation=70)
    if zoom:
        ax.set_ylim([zoom_lower_limit,zoom_upper_limit])
        save_name = f'./plots/graphs/{name}_zoom.png'
    else:
        save_name = f'./plots/graphs/{name}.png'

    plt.suptitle(title)
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


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

    # needed if only running box experiment 
    # df = df.rename(columns={'all': 'box_all'})

    ################################################################
    ###                      CONFIGURATION                       ###
    ###   Experiment - simualted layer activation quantization   ###
    ################################################################

    # Config for plot's text:
    model = 'XL1' # 'L1', 'L2', 'XL0', XL1'
    quant_scheme = 'simulated integer-only' # 'simulated weight-only'
    prompt_type = 'box' #or 'point'

    ################################################################
    ###                      CONFIGURATION                       ###
    ###   Experiment - simualted layer activation quantization   ###
    ################################################################


    df_layer = df.copy()

    df_layer['model'] = df_layer['model'].str.replace('_quant', '')
    df_layer['model'] = df_layer['model'].str.upper()

    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace(model + ':', '')
    df_layer['backbone_version'] = df_layer['backbone_version'].str.replace('stage', '')

    # split data: stage:block:x vs stage:block:layer
    df_layer = df_layer[~df_layer['backbone_version'].str.endswith('x')] # remove blockwise experiments, if any
    

    # plot normal
    plot(
        df_layer,
        title=f'Layer-wise accuracy degredation, {quant_scheme} quantization of {model}',
        xlabel=f'{model} base model with only one layer quantized. Naming scheme is stage:block:layer',
        name=f'{args.file_name}_layer',
        model = model,
        prompt_type=prompt_type,
        add_descriptions=True,
        rotate=True,
        zoom=False,
        )
    
    # plot zoomed
    plot(
        df_layer,
        title=f'Layer-wise accuracy degredation, {quant_scheme} quantization of {model}',
        xlabel=f'{model} base model with only one layer quantized. Naming scheme is stage:block:layer',
        name=f'{args.file_name}_layer',
        model = model,
        prompt_type=prompt_type,
        add_descriptions=True,
        rotate=True,
        zoom=True,
        )
    