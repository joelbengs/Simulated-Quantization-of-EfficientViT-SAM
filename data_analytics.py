import argparse
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from configs.quant_backbones_zoo import REGISTERED_BACKBONE_DESCRIPTIONS_LARGE, REGISTERED_BACKBONE_DESCRIPTIONS_XL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='From pickle file to plots')
    args = parser.parse_args()

    ##########################################
    #                LATENCY                 #
    ##########################################

    data_latency = {
        # Family 1 - red
        'L0 FP16'  : {'model' : 'L0',  'accuracy': 45.7, 'latency': 0.76, 'color': 'red', 'marker' : 'o'},
        'L1 FP16'  : {'model' : 'L1',  'accuracy': 46.2, 'latency': 1.06, 'color': 'red', 'marker' : 'o'},
        'L2 FP16'  : {'model' : 'L2',  'accuracy': 46.6, 'latency': 1.40, 'color': 'red', 'marker' : 'o'},
        'XL0 FP16' : {'model' : 'XL0', 'accuracy': 47.5, 'latency': 1.99, 'color': 'red', 'marker' : 'X'},
        'XL1 FP16' : {'model' : 'XL1', 'accuracy': 47.8, 'latency': 3.75, 'color': 'red', 'marker' : 'X'},
        #'ORIGINAL SAM' : {'model' : 'SAM', 'accuracy': 46.5, 'latency': 61.875, 'color': 'red', 'marker' : 'X'},

        # Family - green
        'L0-Mix-L' :   {'model' : 'L0',  'accuracy': 44.7, 'latency': 0.72, 'color': 'green', 'marker' : 'o'},
        'L1-Mix-L' :   {'model' : 'L1',  'accuracy': 35.3, 'latency': 1.02, 'color': 'green', 'marker' : 'o'},
        'L2-Mix-L' :   {'model' : 'L2',  'accuracy': 29.2, 'latency': 1.34, 'color': 'green', 'marker' : 'o'},
        'XL0-Mix-XL' : {'model' : 'XL0', 'accuracy': 7.8,  'latency': 1.94, 'color': 'green', 'marker' : 'X'},
        'XL1-Mix-XL' : {'model' : 'XL1', 'accuracy': 44.5, 'latency': 3.66, 'color': 'green', 'marker' : 'X'},
#
        ## Family - orange
        #'L0 Mix-MBC-Neck'  : {'model' : 'L0',  'accuracy': 43.8-20, 'latency': 0.82, 'color': 'orange',  'marker' : 'o'},
        #'L1 Mix-MBC-Neck'  : {'model' : 'L1',  'accuracy': 44.0-20, 'latency': 1.12, 'color': 'orange',  'marker' : 'o'},
        #'L2 Mix-MBC-Neck'  : {'model' : 'L2',  'accuracy': 44.1-20, 'latency': 1.44, 'color': 'orange',  'marker' : 'o'},
        #'Xl0 Mix-MBC-Neck' : {'model' : 'XL0', 'accuracy': 10.1-5, 'latency': 2.04, 'color': 'orange',  'marker' : 'X'},
        #'Xl1 Mix-MBC-Neck' : {'model' : 'XL1', 'accuracy': 0.9,  'latency': 3.76, 'color': 'orange',  'marker' : 'X'},
##
        # # Family - blue
        #'L0 Mix-DWSC' :   {'model' : 'L0',  'accuracy': 41.6-20, 'latency': 0.62, 'color': 'blue',  'marker' : 'o'},
        #'L1 Mix-DWSC' :   {'model' : 'L1',  'accuracy': 41.2-20, 'latency': 0.94, 'color': 'blue',  'marker' : 'o'},
        #'L2 Mix-DWSC' :   {'model' : 'L2',  'accuracy': 32.3-20, 'latency': 1.20, 'color': 'blue',  'marker' : 'o'},
        #'Xl0 Mix-DWSC' :  {'model' : 'XL0', 'accuracy': 0.0,  'latency': 1.74, 'color': 'blue',  'marker' : 'X'},
        #'Xl1 Mix-DWSC' :  {'model' : 'XL1', 'accuracy': 0.4,  'latency': 3.56, 'color': 'blue',  'marker' : 'X'},


    }
    
    fig, ax = plt.subplots(figsize=(16,10))
    for name, values in data_latency.items():
        ax.scatter(values['latency'], values['accuracy'], color=values['color'], marker=values['marker'], s=200)  # Increase the size of the dots and add markers
        
        ax.annotate(values['model'], (values['latency'], values['accuracy']+1), color=values['color'], fontsize='xx-large')  # Increase the fontsize and move the annotation up by 0.1 points
        ax.grid(True, linestyle='dotted', alpha=0.5)  # Add gridlines with dotted linestyle and reduced opacity

    # Connect the dots with lines based on color and marker type
    colors = ['red', 'orange', 'blue', 'green']
    for color in colors:
        color_data = {name: values for name, values in data_latency.items() if values['color'] == color}
        markers = set([values['marker'] for values in color_data.values()])
        for marker in markers:
            marker_data = {name: values for name, values in color_data.items() if values['marker'] == marker}
            latencies = [values['latency'] for values in marker_data.values()]
            accuracies = [values['accuracy'] for values in marker_data.values()]
            ax.plot(latencies, accuracies, linestyle='dotted', color=color)


    legend_content = {
        'red' : 'FP16',
        'green' : 'Mix-L/XL',
        'orange': 'Mix-MBC-Neck',
        'blue': 'Mix-DWSC',
    }

    # Add legend
    #legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in legend_content.items()]
    #ax.legend(handles=legend_elements, loc='lower right', fontsize='xx-large')  # Place the legend in the bottom right and increase the fontsize

    plt.text(0.9, 49, 'FP16 Baseline', fontsize=18, color='red', rotation=1.5)
    plt.text(2.5, 49, 'FP16 Baseline', fontsize=18, color='red', rotation=0.5)
    #plt.text(2.5, 48.5, 'FP16 Baseline', fontsize=18, color='red', rotation=0)
    plt.text(0.85, 38, 'Mix-L', fontsize=18, color='green', rotation=-47)
    plt.text(3.0, 32, 'Mix-XL', fontsize=18, color='green', rotation=40)
    plt.text(1.3, 21, 'Mix-MBC-Neck', fontsize=18, color='orange')
    plt.text(1.2, 9, 'Mix-DWSC', fontsize=18, color='blue')

    ax.set_title('Accuracy-latency trade-off with quantization in TensorRT', fontsize='xx-large')
    ax.set_xlabel('TensorRT Latency (ms)', fontsize='xx-large')
    ax.set_ylabel('Zero-Shot COCO mAP', fontsize='xx-large')
 
    ax.set_xlim(0.5,) 
    ax.set_ylim(25,50) 


    name='latency_trade-off-limited'
    plt.savefig(f'./plots/graphs/{name}.png', bbox_inches='tight')
    plt.close()



    ##########################################
    #                Size                    #
    ##########################################

    data_size = {
        # Family 1 - red
        'L0 FP16'  : {'model' : 'L0',  'accuracy': 45.7, 'size': 59, 'color': 'red', 'marker' : 'o'},
        'L1 FP16'  : {'model' : 'L1',  'accuracy': 46.2, 'size': 83, 'color': 'red', 'marker' : 'o'},
        'L2 FP16'  : {'model' : 'L2',  'accuracy': 46.6, 'size': 110, 'color': 'red', 'marker' : 'o'},
        'XL0 FP16' : {'model' : 'XL0', 'accuracy': 47.5, 'size': 215, 'color': 'red', 'marker' : 'X'},
        'XL1 FP16' : {'model' : 'XL1', 'accuracy': 47.8, 'size': 380, 'color': 'red', 'marker' : 'X'},

        # Family 2 - green
        #'L0-Mix-L' :   {'model' : 'L0',  'accuracy': 44.7, 'size': 40  + 30, 'color': 'green', 'marker' : 'o'},
        ##'L1-Mix-L' :   {'model' : 'L1',  'accuracy': 35.3, 'size': 56  + 30, 'color': 'green', 'marker' : 'o'},
        #'L2-Mix-L' :   {'model' : 'L2',  'accuracy': 29.2, 'size': 73  + 30, 'color': 'green', 'marker' : 'o'},
        #'XL0-Mix-XL' : {'model' : 'XL0', 'accuracy': 7.8,  'size': 143 + 30, 'color': 'green', 'marker' : 'X'},
        #'XL1-Mix-XL' : {'model' : 'XL1', 'accuracy': 44.5, 'size': 247 + 30, 'color': 'green', 'marker' : 'X'},

        # Family 3 - blue
        'L0 Mix-DWSC' :   {'model' : 'L0',  'accuracy': 41.6, 'size': 40 , 'color': 'blue',  'marker' : 'o'},
        'L1 Mix-DWSC' :   {'model' : 'L1',  'accuracy': 41.2, 'size': 56 , 'color': 'blue',  'marker' : 'o'},
        'L2 Mix-DWSC' :   {'model' : 'L2',  'accuracy': 32.3, 'size': 73 , 'color': 'blue',  'marker' : 'o'},
        'Xl0 Mix-DWSC' :  {'model' : 'XL0', 'accuracy': 0.0,  'size': 143, 'color': 'blue',  'marker' : 'X'},
        'Xl1 Mix-DWSC' :  {'model' : 'XL1', 'accuracy': 0.4,  'size': 247, 'color': 'blue',  'marker' : 'X'},

        # Family 4 - orange
        'L0 Mix-MBC-Neck'  : {'model' : 'L0',  'accuracy': 43.8, 'size': 52 , 'color': 'orange',  'marker' : 'o'},
        'L1 Mix-MBC-Neck'  : {'model' : 'L1',  'accuracy': 44.0, 'size': 73 , 'color': 'orange',  'marker' : 'o'},
        'L2 Mix-MBC-Neck'  : {'model' : 'L2',  'accuracy': 44.1, 'size': 97 , 'color': 'orange',  'marker' : 'o'},
        'Xl0 Mix-MBC-Neck' : {'model' : 'XL0', 'accuracy': 10.1, 'size': 186, 'color': 'orange',  'marker' : 'X'},
        'Xl1 Mix-MBC-Neck' : {'model' : 'XL1', 'accuracy': 0.9,  'size': 326, 'color': 'orange',  'marker' : 'X'},
    }
    
    fig, ax = plt.subplots(figsize=(16,10))
    for name, values in data_size.items():
        ax.scatter(values['size'], values['accuracy'], color=values['color'], marker=values['marker'], s=200)  # Increase the size of the dots and add markers
        ax.annotate(values['model'], (values['size'], values['accuracy']+0.5), color=values['color'], fontsize='xx-large')  # Increase the fontsize and move the annotation up by 0.1 points
        ax.grid(True, linestyle='dotted', alpha=0.5)  # Add gridlines with dotted linestyle and reduced opacity

    # Connect the dots with lines based on color and marker type
    colors = ['red', 'orange', 'blue', 'green']
    for color in colors:
        color_data = {name: values for name, values in data_size.items() if values['color'] == color}
        markers = set([values['marker'] for values in color_data.values()])
        for marker in markers:
            marker_data = {name: values for name, values in color_data.items() if values['marker'] == marker}
            latencies = [values['size'] for values in marker_data.values()]
            accuracies = [values['accuracy'] for values in marker_data.values()]
            ax.plot(latencies, accuracies, linestyle='dotted', color=color)


    legend_content = {
        'red' : 'FP16',
        #'green' : 'Mix-L/XL',
        'orange': 'Mix-MBC-Neck',
        'blue': 'Mix-DWSC',
    }

    # Add legend
    #legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in legend_content.items()]
    #ax.legend(handles=legend_elements, loc='lower right', fontsize='xx-large')  # Place the legend in the bottom right and increase the fontsize

    plt.text(87, 47, 'FP16 Baseline', fontsize=18, color='red')
    plt.text(75, 43, 'Mix-MBC-Neck', fontsize=18, color='orange')
    plt.text(60, 40, 'Mix-DWSC', fontsize=18, color='blue')

    ax.set_title('Accuracy-size trade-off in simulated quantization', fontsize='xx-large')
    ax.set_xlabel('Size of model (MB)', fontsize='xx-large')
    ax.set_ylabel('Zero-Shot COCO mAP, simulated quantization', fontsize='xx-large')

    #ax.set_xlim(0.5,) 
    ax.set_ylim(0,55) 

    name='size_trade-off'
    plt.savefig(f'./plots/graphs/{name}.png', bbox_inches='tight')
    plt.close()



    ##########################################
    #                Simple                  #
    ##########################################

    data_latency = {
        # Family 1 - red
        'L0 FP16'  : {'model' : 'L0',  'accuracy': 45.7, 'latency': 0.76, 'color': 'red', 'marker' : 'o'},
        'L1 FP16'  : {'model' : 'L1',  'accuracy': 46.2, 'latency': 1.06, 'color': 'red', 'marker' : 'o'},
        'L2 FP16'  : {'model' : 'L2',  'accuracy': 46.6, 'latency': 1.40, 'color': 'red', 'marker' : 'o'},
        'XL0 FP16' : {'model' : 'XL0', 'accuracy': 47.5, 'latency': 1.99, 'color': 'red', 'marker' : 'X'},
        'XL1 FP16' : {'model' : 'XL1', 'accuracy': 47.8, 'latency': 3.75, 'color': 'red', 'marker' : 'X'},
        #'ORIGINAL SAM' : {'model' : 'SAM', 'accuracy': 46.5, 'latency': 61.875, 'color': 'red', 'marker' : 'X'},

        # Family - green
        #'L0-Mix-L' :   {'model' : 'L0',  'accuracy': 44.7, 'latency': 0.72, 'color': 'green', 'marker' : 'o'},
        #'L1-Mix-L' :   {'model' : 'L1',  'accuracy': 35.3, 'latency': 1.02, 'color': 'green', 'marker' : 'o'},
        #'L2-Mix-L' :   {'model' : 'L2',  'accuracy': 29.2, 'latency': 1.34, 'color': 'green', 'marker' : 'o'},
        #'XL0-Mix-XL' : {'model' : 'XL0', 'accuracy': 7.8,  'latency': 1.94, 'color': 'green', 'marker' : 'X'},
        #'XL1-Mix-XL' : {'model' : 'XL1', 'accuracy': 44.5, 'latency': 3.66, 'color': 'green', 'marker' : 'X'},
#
        ## Family - orange
        #'L0 Mix-MBC-Neck'  : {'model' : 'L0',  'accuracy': 43.8-20, 'latency': 0.82, 'color': 'orange',  'marker' : 'o'},
        #'L1 Mix-MBC-Neck'  : {'model' : 'L1',  'accuracy': 44.0-20, 'latency': 1.12, 'color': 'orange',  'marker' : 'o'},
        #'L2 Mix-MBC-Neck'  : {'model' : 'L2',  'accuracy': 44.1-20, 'latency': 1.44, 'color': 'orange',  'marker' : 'o'},
        #'Xl0 Mix-MBC-Neck' : {'model' : 'XL0', 'accuracy': 10.1-5, 'latency': 2.04, 'color': 'orange',  'marker' : 'X'},
        #'Xl1 Mix-MBC-Neck' : {'model' : 'XL1', 'accuracy': 0.9,  'latency': 3.76, 'color': 'orange',  'marker' : 'X'},
#
       ## # Family - blue
        #'L0 Mix-DWSC' :   {'model' : 'L0',  'accuracy': 41.6-20, 'latency': 0.62, 'color': 'blue',  'marker' : 'o'},
        #'L1 Mix-DWSC' :   {'model' : 'L1',  'accuracy': 41.2-20, 'latency': 0.94, 'color': 'blue',  'marker' : 'o'},
        #'L2 Mix-DWSC' :   {'model' : 'L2',  'accuracy': 32.3-20, 'latency': 1.20, 'color': 'blue',  'marker' : 'o'},
        #'Xl0 Mix-DWSC' :  {'model' : 'XL0', 'accuracy': 0.0,  'latency': 1.74, 'color': 'blue',  'marker' : 'X'},
        #'Xl1 Mix-DWSC' :  {'model' : 'XL1', 'accuracy': 0.4,  'latency': 3.56, 'color': 'blue',  'marker' : 'X'},


    }
    
    fig, ax = plt.subplots(figsize=(16,10))
    for name, values in data_latency.items():
        ax.scatter(values['latency'], values['accuracy'], color=values['color'], marker=values['marker'], s=300)  # Increase the size of the dots and add markers
        ax.annotate(values['model'], (values['latency'], values['accuracy']+0.5), color=values['color'], fontsize=18)  # Increase the fontsize and move the annotation up by 0.1 points
        ax.grid(True, linestyle='dotted', alpha=0.5)  # Add gridlines with dotted linestyle and reduced opacity

    # Connect the dots with lines based on color and marker type
    colors = ['red', 'orange', 'blue', 'green']
    for color in colors:
        color_data = {name: values for name, values in data_latency.items() if values['color'] == color}
        markers = set([values['marker'] for values in color_data.values()])
        for marker in markers:
            marker_data = {name: values for name, values in color_data.items() if values['marker'] == marker}
            latencies = [values['latency'] for values in marker_data.values()]
            accuracies = [values['accuracy'] for values in marker_data.values()]
            ax.plot(latencies, accuracies, linestyle='dotted', color=color)


    legend_content = {
        'red' : 'FP16',
        'green' : 'Mix-L/XL',
        'orange': 'Mix-MBC-Neck',
        'blue': 'Mix-DWSC',
    }

    # Add legend
    #legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for color, label in legend_content.items()]
    #ax.legend(handles=legend_elements, loc='lower right', fontsize='xx-large')  # Place the legend in the bottom right and increase the fontsize

    plt.text(0.75, 44.5, 'EfficientViT-SAM L-Series', fontsize=24, color='red', rotation=15)
    plt.text(2.3, 46.5, 'EfficientViT-SAM XL-Series', fontsize=24, color='red', rotation=2)
    #plt.text(2.5, 48.5, 'FP16 Baseline', fontsize=18, color='red', rotation=0)
    #plt.text(0.85, 38, 'Mix-L', fontsize=18, color='green', rotation=-47)
    #plt.text(3.0, 32, 'Mix-XL', fontsize=18, color='green', rotation=40)
    #plt.text(75, 43, 'Mix-MBC-Neck', fontsize=18, color='orange')
    #plt.text(60, 40, 'Mix-DWSC', fontsize=18, color='blue')

    # ax.set_title('Accuracy-latency trade-off with quantization in TensorRT', fontsize='xx-large')
    # ax.set_xlabel('TensorRT Latency (ms)', fontsize='xx-large')
    # ax.set_ylabel('Zero-Shot COCO mAP', fontsize='xx-large')

    ax.set_title('Accuracy-latency trade-off', fontsize=24)
    ax.set_xlabel('Latency (ms)', fontsize=24)
    ax.set_ylabel('Accuracy (mAP)', fontsize=24)

    ax.set_xlim(0,4) 
    ax.set_ylim(40,50) 


    name='method_graph'
    plt.savefig(f'./plots/graphs/{name}.png', bbox_inches='tight')
    plt.close()
