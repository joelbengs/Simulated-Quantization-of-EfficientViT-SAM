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
    zoom_lower_limit=77.1
    zoom_upper_limit = baseline_value + 0.2
    #if zoom:
    #    zoom_upper_limit = 80
    #else:
    #    zoom_upper_limit = baseline_value + 0.2
    texts = []
    
    if prompt_type == 'box' or prompt_type == 'point':
        performance_measure = 'all'
    elif performance_measure == 'point_and_box':
        raise KeyError("plot function that can plot both point and box in one call not yet implemented")
    
    fig, ax = plt.subplots(figsize=(25,5))
    ax.plot(df['backbone_version'], df[performance_measure])
    ax.scatter(df['backbone_version'], df[performance_measure])


    if add_descriptions:
        # Add description to each data point where 'all' score is substantially lower than baseline
        for i, val in df.iterrows(): # index and row data
            if zoom:
                if val[performance_measure] < baseline_value-0.4: # L2: 0.3
                    # using adjustText library to avoid overlapping text
                    texts.append(ax.text(val['backbone_version'], val['adjusted_performance_measure'] - 0.1, f"{description_dict[val['backbone_version']]}"))
            else:
                if val[performance_measure] < baseline_value-4:
                    #ax.text(val['backbone_version'], val[performance_measure], f"{description_dict[val['backbone_version']]}")
                    texts.append(ax.text(val['backbone_version'], val[performance_measure] - 0.1, f"{description_dict[val['backbone_version']]}"))
        adjust_text(texts)  # adjust text to minimize overlaps

        
    # vertical lines at each x mark
    for x in df['backbone_version']:
        ax.axvline(x, color='lightgray', linestyle='dotted')
    ax.axhline(baseline_value, color='red', linestyle='dotted')
    #ax.text(0, baseline_value - 0.1 if zoom else baseline_value - 1, 'Baseline', va='top', ha="right")

    if zoom: ax.set_xlabel(xlabel)
    if not zoom: plt.suptitle(title)
    ax.set_ylabel('mIoU (higher is better)')
    if rotate:
        plt.xticks(rotation=70)
    if zoom:
        ax.set_ylim([zoom_lower_limit,zoom_upper_limit])
        save_name = f'./plots/graphs/{name}_zoom.png'
    else:
        ax = plt.gca()  # get current axes
        ymin, _ = ax.get_ylim()  # get the current lower limit
        ax.set_ylim(ymin, zoom_upper_limit)  # set the new upper limit
        save_name = f'./plots/graphs/{name}.png'

    plt.savefig(save_name, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='From pickle file to plots')
    parser.add_argument('--file_path', type=str, default='results', help='The directory of the pickle file')
    parser.add_argument('--file_name', type=str, help='The name of the pickle file to analyse')
    parser.add_argument('--model', type=str, default='L0',  help='The model of the test, in capital letters')
    args = parser.parse_args()

    # load the data from storage
    df, _ = read_pickle_to_dataframe(args.file_path, args.file_name)

    pd.set_option('display.max_rows', None)
    print("data analytics script is working with the following data:")
    print(df.columns)

    ################################################################
    ###                      CONFIGURATION                       ###
    ###   Experiment - simualted layer activation quantization   ###
    ################################################################

    # Config for plot's text:
    # model = 'L0' # 'L1', 'L2', 'XL0', XL1'
    model = args.model
    quant_scheme = 'simulated integer-only (weights + activations)' # 'simulated weight-only'
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
    