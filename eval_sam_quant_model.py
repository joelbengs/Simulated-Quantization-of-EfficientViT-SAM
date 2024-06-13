# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented simulation of mixed-precision quantization to further accelerate EfficientViT-SAM

import argparse
import json
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from efficientvit.quant_backbones_zoo import REGISTERED_BACKBONE_VERSIONS
from efficientvit.models.nn.ops import QConvLayer, QConvLayerV2, QLiteMLA
from efficientvit.models.ptq.bit_type import BitType
from efficientvit.models.ptq.observer.base import BaseObserver
from lvis import LVIS
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchinfo import summary

from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.sam_model_zoo import create_sam_model
from sam_eval_utils import Clicker, evaluate_predictions_on_coco, evaluate_predictions_on_lvis, get_iou_metric, iou
from quant_config import Config


def bbox_xywh_to_xyxy(bbox: list[int]) -> list[int]:
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def ann_to_mask(ann, h, w):
    if type(ann["segmentation"]) == list:
        rles = mask_util.frPyObjects(ann["segmentation"], h, w)
        rle = mask_util.merge(rles)
    elif type(ann["segmentation"]["counts"]) == list:
        rle = mask_util.frPyObjects(ann["segmentation"], h, w)
    else:
        raise NotImplementedError()

    mask = mask_util.decode(rle) > 0

    return mask


def sync_output(world_size, output):
    outs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(outs, output)
    merged_outs = []
    for sublist in outs:
        merged_outs += sublist

    return merged_outs


def predict_mask_from_box(predictor: EfficientViTSamPredictor, bbox: np.ndarray) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


def predict_mask_from_point(
    predictor: EfficientViTSamPredictor, point_coords: np.ndarray, point_labels: np.ndarray
) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask


class eval_dataset(Dataset):
    def __init__(self, dataset, image_root, prompt_type, annotation_json_file, source_json_file=None):
        self.dataset = dataset
        self.image_root = image_root
        self.prompt_type = prompt_type
        self.annotation_json_file = annotation_json_file

        if self.dataset == "coco":
            self.images = os.listdir(self.image_root)
            self.images = [os.path.join(self.image_root, image) for image in self.images]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
        elif self.dataset == "lvis":
            self.images = json.load(open(self.annotation_json_file, "r"))["images"]
            self.images = [
                os.path.join(self.image_root, image["coco_url"].split("/")[-2], image["coco_url"].split("/")[-1])
                for image in self.images
            ]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
        else:
            raise NotImplementedError()

        if self.prompt_type == "point" or self.prompt_type == "box":
            self.annotations = json.load(open(self.annotation_json_file, "r"))["annotations"]
        elif self.prompt_type == "box_from_detector":
            self.source_json_file = json.load(open(source_json_file))
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.prompt_type == "point" or self.prompt_type == "box":
            anns = [ann for ann in self.annotations if ann["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "anns": anns}
        elif self.prompt_type == "box_from_detector":
            detections = [det for det in self.source_json_file if det["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "detections": detections}
        else:
            raise NotImplementedError()


class calib_dataset(Dataset):
    """
    A dataset class for calibration of quantization operators. It accepts coco and lvis datasets, but note that the model should not calibrate on the same dataset as validation in the zero-shot task.

    Args:
        dataset (str): The name of the dataset. Supported values are "sa-1b", "coco", and "lvis".
        image_root (str): The root directory of the images.
        prompt_type (str): The type of prompt. Supported values are "point", "box", and "box_from_detector".
        annotation_json_file (str): The path to the annotation JSON file.
        source_json_file (str, optional): The path to the source JSON file. Required only if prompt_type is "box_from_detector".

    Raises:
        NotImplementedError: If calibrating using a dataset other than "sa-1b", "coco", or "lvis", the naming scheme must be defined. Make sure to provide the --dataset_calibration argument.

    Attributes:
        images (list): A list of image paths.
        ids (list): A list of image IDs.

    Note:
        A calibration dataloader should not use annotations, but they are included here to make other functions happy.
    """

    def __init__(self, dataset, image_root, prompt_type, annotation_json_file, source_json_file=None):
        self.dataset = dataset
        self.image_root = image_root
        self.prompt_type = prompt_type
        self.annotation_json_file = annotation_json_file

        if self.dataset == "sa-1b":
            self.images = os.listdir(self.image_root)
            self.images = [os.path.join(self.image_root, image) for image in self.images]
            self.ids = [int(image.split("/")[-1].split(".")[0].replace('sa_', '')) for image in self.images]
        elif self.dataset == "coco":
            self.images = os.listdir(self.image_root)
            self.images = [os.path.join(self.image_root, image) for image in self.images]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images] # assumes image names on the form "000012345.jpg"
        elif self.dataset == "lvis":
            self.images = json.load(open(self.annotation_json_file, "r"))["images"]
            self.images = [
                os.path.join(self.image_root, image["coco_url"].split("/")[-2], image["coco_url"].split("/")[-1])
                for image in self.images
            ]
            self.ids = [int(image.split("/")[-1].split(".")[0]) for image in self.images]
        else:
            raise NotImplementedError("If calibrating using other dataset than SA-1B, coco, or lvis, you must define the naming scheme. Did you forget the argument --dataset_calibration?")
        # A calibration dataloader should not use annotations, but here they are to make other funcitons happy
        if self.prompt_type == "point" or self.prompt_type == "box":
            self.annotations = json.load(open(self.annotation_json_file, "r"))["annotations"]
        elif self.prompt_type == "box_from_detector":
            self.source_json_file = json.load(open(source_json_file))
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        if self.prompt_type == "point" or self.prompt_type == "box":
            anns = [ann for ann in self.annotations if ann["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "anns": anns}
        elif self.prompt_type == "box_from_detector":
            detections = [det for det in self.source_json_file if det["image_id"] == self.ids[idx]]
            return {"image_path": image_path, "detections": detections}
        else:
            raise NotImplementedError()

 
def collate_fn(batch):
    return batch


def toggle_operation(efficientvit_sam, operation, state, backbone_version: str, print_progress=False):
    """
    This function is used to toggle a specific operation on or off in the EfficientViT SAM model.

    Parameters:
    efficientvit_sam (object): The EfficientViT SAM model in which the operation needs to be toggled.
    operation (str): The name of the operation to be toggled.
    state (str): The state to which the operation needs to be toggled. Must be either 'on' or 'off'.
    backbone_version (str): The version of the backbone used in the model. Must be one of the registered backbone versions from quant_backbone_zoo.py.
    print_progress (bool, optional): If set to True, progress will be printed. Defaults to False.

    The function works as follows:
    1. Checks if the state is either 'on' or 'off'. If not, raises a ValueError.
    2. Checks if the backbone_version is one of the registered versions. If not, raises a NotImplementedError.
    3. Calls the appropriate method on the model to toggle the operation to the desired state.

    Note: The specific method called on the model is determined by the operation and state parameters. 
    For example, if operation is 'calibrate' and state is 'on', the method called would be 'toggle_selective_calibrate_on'.
    """

    printout=(local_rank==0 and print_progress is True)
    if state not in ['on', 'off']:
        raise ValueError("State must be either 'on' or 'off'")
    if backbone_version in REGISTERED_BACKBONE_VERSIONS:
        getattr(efficientvit_sam, f'toggle_selective_{operation}_{state}')(printout=printout, **REGISTERED_BACKBONE_VERSIONS[backbone_version])
    else:
        raise NotImplementedError("Backbone version not yet implemented")


def calibrate_image_encoder(efficientvit_sam, calib_dataloader, args, local_rank):
    """
    This function is used to calibrate the quantization operators.

    Parameters:
    efficientvit_sam (object): The EfficientViT SAM model that needs to be calibrated.
    calib_dataloader (DataLoader): The DataLoader object that provides batches of images for calibration.
    args (object): An object containing various arguments needed for the calibration process. 
                   This includes 'backbone_version' and 'limit_iterations'.
    local_rank (int): The rank of the current process in a distributed setting. Used to avoid double printouts.

    The function works as follows:
    1. Moves the model to the correct GPU and sets it to evaluation mode.
    2. Creates a predictor using the model.
    3. Turns on the 'calibrate' operation for all relevant modules in the model.
    4. Iterates over each batch of images from the DataLoader.
    5. If it's the second to last batch, it turns on the 'last_calibrate' operation for all relevant modules, so that they can fetch their parameters in the last batch.
    6. For each image, it runs inference through the image encoder.
    7. After all images have been processed, it turns off the 'calibrate' and 'last_calibrate' operations.

    Note: The length of the DataLoader is dynamic as data is split over GPUs.
    """
    printout=(local_rank==0)
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()                 # move model to correct GPU, and turn on eval mode
    predictor = EfficientViTSamPredictor(efficientvit_sam)                      # create predictor
    toggle_operation(efficientvit_sam, 'calibrate', 'on', args.backbone_version, print_progress=False)                       # sets calibrate = true for all 'relevant' modules

    for i, data in enumerate(tqdm(calib_dataloader, disable=local_rank != 0)):  # for each batch of images
        if i == len(calib_dataloader) - 1 or i == args.limit_iterations - 1:    # The lenght of the dataloader is dynamicas data is split over GPUs. zero-based enumeration              
            toggle_operation(efficientvit_sam, 'last_calibrate', 'on', args.backbone_version, print_progress=False)          # if second to last batch, set last_calibrate = true for all relevant modules
        if i == args.limit_iterations:
            break
        data = data[0]                                                          # fetch the images?
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))     # convert ot RGB image
        predictor.set_image(sam_image)                                          # this call runs inference through the image encoder!

    toggle_operation(efficientvit_sam, 'calibrate', 'off', args.backbone_version, print_progress=False)
    toggle_operation(efficientvit_sam, 'last_calibrate', 'off', args.backbone_version, print_progress=False)   

def run_box(efficientvit_sam, dataloader, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()                 # move model to correct GPU, and turn on eval mode
    predictor = EfficientViTSamPredictor(efficientvit_sam)                      # create predictor

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):        # for each batch of images
        data = data[0]                                                          # fetch the images?
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))     # convert ot RGB image
        predictor.set_image(sam_image)                                          # this call runs inference through the image encoder!
        anns = data["anns"]                                                     # fetch annotations for the batch
        for ann in anns:                                                        # for each annotation
            if ann["area"] < 1:                                                 # skip if too small bounding box
                continue

            sam_mask = ann_to_mask(ann, sam_image.shape[0], sam_image.shape[1]) # find true mask

            bbox = np.array(bbox_xywh_to_xyxy(ann["bbox"]))                     # find bounding box - (i.e. the prompt)
            pre_mask = predict_mask_from_box(predictor, bbox)                   # predict mask from bounding box

            miou = iou(pre_mask, sam_mask)                                      # compare prediction to true mask

            result = {
                "area": ann["area"],
                "iou": miou,
            }

            output.append(result)

    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output)                              # synchronice all processes and merge results

    return merged_outs

def run_point(efficientvit_sam, dataloader, num_click, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = EfficientViTSamPredictor(efficientvit_sam)

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))
        predictor.set_image(sam_image)
        anns = data["anns"]
        for ann in anns:
            if ann["area"] < 1:
                continue

            sam_mask = ann_to_mask(ann, sam_image.shape[0], sam_image.shape[1])

            point_coords_list = []
            point_labels_list = []

            clicker = Clicker(gt_mask=sam_mask)
            pre_mask = np.zeros_like(sam_mask)

            for i in range(num_click):
                clicker.make_next_click(pre_mask)
                point_coords_list.append(clicker.clicks_list[-1].coords[::-1])
                point_labels_list.append(int(clicker.clicks_list[-1].is_positive))
                point_coords = np.stack(point_coords_list, axis=0)
                point_labels = np.array(point_labels_list)

                pre_mask = predict_mask_from_point(predictor, point_coords, point_labels)

            miou = iou(pre_mask, sam_mask)

            result = {
                "area": ann["area"],
                "iou": miou,
            }

            output.append(result)

    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output)

    return merged_outs

def run_box_from_detector(efficientvit_sam, dataloader, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()
    predictor = EfficientViTSamPredictor(efficientvit_sam)

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):
        data = data[0]
        sam_image = Image.open(data["image_path"]).convert("RGB")
        predictor.set_image(np.array(sam_image))
        detections = data["detections"]
        for det in detections:
            bbox = np.array(bbox_xywh_to_xyxy(det["bbox"]))
            sam_mask = predict_mask_from_box(predictor, bbox)
            rle = mask_util.encode(np.array(sam_mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            det["segmentation"] = rle
        output += detections
    world_size = int(os.environ["WORLD_SIZE"])
    merged_outs = sync_output(world_size, output)

    return merged_outs

# Evaluates a model with printout
def evaluate(results, prompt_type, dataset, annotation_json_file=None):
    if prompt_type == "point" or prompt_type == "box":
        print(", ".join([f"{key}={val:.3f}" for key, val in get_iou_metric(results).items()]))
    elif prompt_type == "box_from_detector":
        iou_type = "segm"
        if dataset == "coco":
            coco_api = COCO(annotation_json_file)
            evaluate_predictions_on_coco(coco_gt=coco_api, coco_results=results, iou_type=iou_type)
        elif dataset == "lvis":
            lvis_api = LVIS(annotation_json_file)
            evaluate_predictions_on_lvis(lvis_gt=lvis_api, lvis_results=results, iou_type=iou_type)
    else:
        raise NotImplementedError()

# Evaluates a model to a pandas dataframe
def evaluate_to_dataframe(dataframe, results, prompt_type, dataset, annotation_json_file=None, args=None):
    # append each new result to the dataframe, and return the dataframe
    if prompt_type == "point" or prompt_type == "box":
        metrics = get_iou_metric(results)
        for key, val in metrics.items():
            dataframe.at[dataframe.index[-1], key] = val
        more_metrics = calculate_savings(efficientvit_sam=efficientvit_sam)
        for key, val in more_metrics.items():
            dataframe.at[dataframe.index[-1], key] = val
        return dataframe
    elif prompt_type == "box_from_detector":
        iou_type = "segm"
        if dataset == "coco":
           raise NotImplementedError()
        elif dataset == "lvis":
            raise NotImplementedError()
    else:
        raise NotImplementedError()

# Creates or loads a pandas dataframe
def create_dataframe(prompt_type, columns, script_name: str) -> pd.DataFrame:
    # add columns
    if prompt_type == 'box' or prompt_type == "point":
            columns.extend([
            "all",
            "large",
            "medium",
            "small", 
            ])
    elif prompt_type == "box_from_detector":
        raise NotImplementedError("create_dataframe not implemented for prompt_type")
    else:
        raise NotImplementedError("create_dataframe not implemented for prompt_type")
    
    # fetch existing dataframe, else create new dataframe
    # remove any leading directories and extensions from the script name
    script_name = os.path.basename(script_name)
    script_name = os.path.splitext(script_name)[0]
    file_path = f'results/{script_name}.pkl'
    
    if os.path.exists(file_path):
        df = pd.read_pickle(file_path)
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan
    else:
        df = pd.DataFrame(columns=columns)
    return df

# Saves metadata about the current experiment to a dataframe
def metadata_to_dataframe(dataframe: pd.DataFrame, args, config, columns) -> pd.DataFrame:
    row_data = {}
    # save metadata from args
    for column in columns:
        row_data[column] = getattr(args, column)

    # fetch the information in the BitType objects, instead of just object references
    config_as_dict = vars(config).copy()
    for key, value in config_as_dict.items():
        if isinstance(value, BitType):
            config_as_dict[key] = value.to_dict()

    # save metadata from quant config. overwrites the previous loop if conflicting
    for key,val in config_as_dict.items():
        row_data[key] = val

    # concatenate row_data to existing dataframe
    dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    return dataframe

# Saves pandas dataframe to pkl file
def save_dataframe_to_file(dataframe: pd.DataFrame, script_name: str) -> None:
    # Save the dataframe as a pickle file in the 'results' directory. Will overwrite.
    # remove any leading directories and extensions from the script name
    script_name = os.path.basename(script_name)
    script_name = os.path.splitext(script_name)[0]
    dataframe.to_pickle(path=f'results/{script_name}.pkl')

# Calculates the megabytes saved from quantization
def calculate_savings(efficientvit_sam):
    # calculates the theorethical savings from the applied quantization. Assumes from FP16 to INT8
    affected = efficientvit_sam.get_number_of_quantized_params() #number of weights
    bytes_saved = affected #as Int8 saves 8 bits = 1 byte per weight compared to the memory requirement of FP16
    megabytes_saved = affected/1024/1024
    # print(f"quantized {affected} params to int8, \nsaving {megabytes_saved:.2f} Mb compared to FP16.")
    print(f'saved {megabytes_saved:.2f} Mb')
    return {
        "number of quantized params": affected,
        "bytes_saved": bytes_saved,
        "megabytes_saved": megabytes_saved,
    }

# helper method that plots the historam of an individual observer
def plot_histogram_of_observer(observer: BaseObserver, sizes, model: str):
        if observer.stored_tensor.numel() == 0:
            print(f"The tensor of {observer.stage_id}:{observer.block_position}:{observer.layer_position}:{observer.weight_norm_or_act} observer is empty! Has calibration been performed with attribute monitor_distributions turned on?")

        tensor = observer.stored_tensor.clone() # might need to clone to isolate from other processes
        tensor = tensor.detach() # can't call numpy() on tensor that requires grad
        tensor = tensor.cpu() # torch.histogram is not implemented on CUDA backend.

        hist_values, bin_edges = torch.histogram(tensor, density=True) # density calculates the density distributions isntead of just number of occurances

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(bin_edges[:-1], hist_values, width = 0.1)
        ax.set_title(f"Distribution of {observer.weight_norm_or_act} of {observer.stage_id}:{observer.block_position}:{observer.layer_position} - {observer.block_name}, \n tensor shape {tensor.size()}, 25 calibration samples")
        
        print(f"Plotting {observer.stage_id}:{observer.block_position}:{observer.layer_position}:{observer.weight_norm_or_act} with tensor size {tensor.size()}, numel = {tensor.numel()}")
        sizes[f"{observer.stage_id}:{observer.block_position}:{observer.layer_position}:{observer.weight_norm_or_act}:{observer.block_name}"] = sci_num = format(tensor.numel(), '.2e')

        # plot mean and standard deviation
        mean = torch.mean(tensor).item()
        std_dev = torch.std(tensor).item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        ax.axvline(mean, color='b', linestyle='dotted', linewidth=1, label=f'mean = {round(mean, 3)}')
        ax.axvline(mean - std_dev, color='r', linestyle='dashed', linewidth=1, label=f'mean - std_dev = {round(mean - std_dev, 3)}')  # mean - std_dev
        ax.axvline(mean + std_dev, color='r', linestyle='dashed', linewidth=1, label=f'mean + std_dev = {round(mean + std_dev, 3)}')  # mean + std_dev

        ax.axvline(min_val, color='g', linestyle='dotted', linewidth=1, label=f'min = {round(min_val, 3)}')
        ax.axvline(max_val, color='g', linestyle='dotted', linewidth=1, label=f'max = {round(max_val, 3)}')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        
        ax.set_xlabel("Weight value")
        ax.set_ylabel(f"Relative Frequency of pre-trained weights" if observer.weight_norm_or_act == 'weight' else "Relative Frequency after calibration")
        fig.tight_layout()
        plt.savefig(f'./plots/histograms/{model.split("_")[0]}/histogram_{observer.stage_id}:{observer.block_position}:{observer.layer_position}_{observer.block_name}_{observer.weight_norm_or_act}.png')
        plt.close()

# Function that calls plot_histogram_of_observer for all the observers in the model's image encoder
def plot_distributions_of_image_encoder(efficientvit_sam, model: str):
    '''
    This function plots the distributions of weights, activations and norms of a given model.
    The results are saved under .plots/histograms
    This feature is experimental and should be used with caution.

    This functinoality is implemented using observers. Each layer has one observer object connected to each of its operatiions (conv, norm, act...).
    When the attribute "monitor_distirbutions" is toggled to True, the observer object will store all sample tensors passing through during calibration.
    Weights are static, so more than one pass of calibration will not alter the weight distributions.
    Tensors passing throgh norm and activation will be concatenated in the observer. There is a risk for memory overflow, in which case the tensors should be reduced to histogram representations before storage.
    However, this is not implemented, so do limit the iterations in calibration using the argument --limit_iterations 100.

    After calibration, the tensor stored in each observer is processed into a histogram and exported with matplotlib.

    To plot the  call the main method with:
    
    --model l0_quant                            or any other quant model. only models using a quantizable backbone and neck can be analyzed
    --backbone_version (any)                    Monitoring occurs during calibration, not inference, so nothing is quantized
    --plot_distributions                        this will trigger this functionality

    '''
    # assuming efficientvit_sam.toggle_monitor_distributions_on() has been called before calibration, and calibration has been run for at least one sample.
    sizes_w = {}
    sizes_a = {}
    sizes_n = {}
    for m in efficientvit_sam.image_encoder.modules():
        if type(m) in [QConvLayer, QConvLayerV2, QLiteMLA]:
            if hasattr(m, 'weight_observer'):
                plot_histogram_of_observer(m.weight_observer, sizes_w, model)
            if hasattr(m, 'act_observer'):
                plot_histogram_of_observer(m.act_observer, sizes_a, model)
            if hasattr(m, 'norm_observer'):
                plot_histogram_of_observer(m.norm_observer, sizes_n, model)
    print("size of weight tensors")
    for key in sizes_w.keys():
        print(key, sizes_w[key])
    print("size of act tensors")
    for key in sizes_a.keys():
        print(key, sizes_a[key])
    print("size of norm tensors")
    for key in sizes_n.keys():
        print(key, sizes_n[key])
    efficientvit_sam.toggle_monitor_distributions_off()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="point", choices=["point", "box", "box_from_detector"])
    parser.add_argument("--num_click", type=int, default=1)
    parser.add_argument("--dataset", type=str, choices=["coco", "lvis"])
    parser.add_argument("--dataset_calibration", type=str, choices=["sa-1b", "coco", "lvis"], default="sa-1b")
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--image_root_calibration", type=str, default="sa-1b")
    parser.add_argument("--annotation_json_file", type=str)
    parser.add_argument("--source_json_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--single_gpu', action='store_true', help="Force the use of a single gpu, might help in troubleshooting quantization")
    parser.add_argument('--print_progress', action="store_true", help="shows debugging printouts")
    parser.add_argument('--export_dataframe', action="store_true")
    parser.add_argument('--script_name', type=str)
    parser.add_argument('--limit_iterations', type=int, default=2500, help="How many calibration samples to use at the maximum, per GPU") 
    parser.add_argument('--print_torchinfo', action='store_true', help="printouts information about the PyTorch model. Adjust depth in the main method")

    parser.add_argument("--quantize_W", action="store_true", help="Turn on quantization and calibration for weights")
    parser.add_argument("--quantize_A", action="store_true", help="Turn on quantization and calibration for activations")
    parser.add_argument("--quantize_N", action="store_true", help="Turn on quantization and calibration for norms (Warning: experimental feature)")

    parser.add_argument("--observer_method_W", choices=["minmax", "ema", "omse", "percentile"])
    parser.add_argument("--observer_method_A", choices=["minmax", "ema", "omse", "percentile"])
    parser.add_argument("--observer_method_N", choices=["minmax", "ema", "omse", "percentile"])

    parser.add_argument("--quantize_method_W", choices=["uniform", "log2"])
    parser.add_argument("--quantize_method_A", choices=["uniform", "log2"])
    parser.add_argument("--quantize_method_N", choices=["uniform", "log2"])

    parser.add_argument("--calibration_mode_W", choices=["layer_wise", "channel_wise"])
    parser.add_argument("--calibration_mode_A", choices=["layer_wise", "channel_wise"])
    parser.add_argument("--calibration_mode_N", choices=["layer_wise", "channel_wise"])

    parser.add_argument("--plot_distributions", action="store_true", help="monitors and plots distributions of weiights and activations. Must be used with _quant model and should be used with an FP32 backbone")
    parser.add_argument("--backbone_version", type=str, help="backbones are defined in quant_backbones_zoo.py")
    args = parser.parse_args()

    # Quantization details are built stored in a Config object, passed into the model
    quant_config = Config(args)

    # colums for dataframes when running scripts.
    columns = [
        "model",
        "backbone_version",
        "prompt_type",
        "quantize_W",
        "quantize_A",
        "quantize_N",
        "num_click",
        "dataset",
        "dataset_calibration",
        "image_root",
        "image_root_calibration",
        "limit_iterations",
        "annotation_json_file",
        "source_json_file",
    ]

    if args.single_gpu:
        local_rank = 0
        if local_rank == 0 and not args.suppress_print: # only master process prints
            print("Using single GPU")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl") # initializing the distributed environment 
        if local_rank == 0 and args.print_progress:
            print(f"Using {torch.distributed.get_world_size()} GPUs")
    torch.cuda.set_device(local_rank)

    # Override the built-in print function so only master process prints
    def print(*args, **kwargs):
        if local_rank == 0:
            __builtins__.print(*args, **kwargs)

    if args.quantize_N:
        print("Warning: Quant of norms is turned on, which is an experimental feature")

    # model creation
    efficientvit_sam = create_sam_model(name=args.model, pretrained=True, weight_url=args.weight_url, config=quant_config)

    if args.print_torchinfo and local_rank == 0:
        # Use torchinfo.summary to print the model. Depth controls granularity of printout - all params are counted anyhow
        summary(
            efficientvit_sam.image_encoder, 
            depth=5,
            col_names = ("output_size", "num_params", "mult_adds"),
            input_size=(1, 3, 2014, 512)
            )

    # dataset creation
    dataset = eval_dataset(args.dataset, args.image_root, args.prompt_type, args.annotation_json_file, args.source_json_file)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn)
    if args.print_progress:
        print(f"The dataloader contains {len(dataloader.dataset)} images from directory {args.dataset}.")

    # calibration dataset
    if args.quantize_W or args.quantize_A or args.quantize_N or args.plot_distributions:
        calib_dataset = calib_dataset(args.dataset_calibration, args.image_root_calibration, args.prompt_type, args.annotation_json_file, args.source_json_file)
        calib_sampler = DistributedSampler(calib_dataset, shuffle=False)
        calib_dataloader = DataLoader(calib_dataset, batch_size=1, sampler=calib_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn)
        if args.print_progress:
            print(f"The calibration dataloader contains {len(calib_dataloader.dataset)} images from directory {args.dataset_calibration}.")
        if args.plot_distributions and local_rank == 0:
            efficientvit_sam.toggle_monitor_distributions_on()

    # run calibration and toggle quantization on
    if args.quantize_W or args.quantize_A or args.quantize_N:
        if args.print_progress:
            print(f"Calibrating image encoder using {args.limit_iterations} samples")
        calibrate_image_encoder(efficientvit_sam, calib_dataloader, args, local_rank)
        if args.quantize_W:
            toggle_operation(efficientvit_sam, 'quant_weights', 'on', args.backbone_version, args.print_progress)
        if args.quantize_A:
            toggle_operation(efficientvit_sam, 'quant_activations', 'on', args.backbone_version, args.print_progress)
        if args.quantize_N:
            toggle_operation(efficientvit_sam, 'quant_norms', 'on', args.backbone_version, args.print_progress) # experimental feature

    if args.plot_distributions and local_rank == 0:
        plot_distributions_of_image_encoder(efficientvit_sam, model=args.model)

    # inference
    if args.prompt_type == "point":
        results = run_point(efficientvit_sam, dataloader, args.num_click, local_rank)
    elif args.prompt_type == "box":
        results = run_box(efficientvit_sam, dataloader, local_rank)
    elif args.prompt_type == "box_from_detector":
        results = run_box_from_detector(efficientvit_sam, dataloader, local_rank)
    else:
        raise NotImplementedError(f"The task {args.prompt_type} is not implemented")

    # evaluation - only done my the master process, not other parallell processes
    if local_rank == 0:
        if args.export_dataframe:
            df = create_dataframe(args.prompt_type, columns.copy(), args.script_name)
            df = metadata_to_dataframe(df, args, quant_config, columns)        
            df = evaluate_to_dataframe(df, results, args.prompt_type, args.dataset, args.annotation_json_file, args=args)
            print("New row added to results: \n", df.tail(1))
            save_dataframe_to_file(df, args.script_name)
        else:
            print("")
            evaluate(results, args.prompt_type, args.dataset, args.annotation_json_file)
            calculate_savings(efficientvit_sam)
