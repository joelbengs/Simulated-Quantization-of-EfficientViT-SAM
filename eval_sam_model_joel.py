
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from lvis import LVIS
from PIL import Image
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

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


def collate_fn(batch):
    return batch

# Given a bounding box, let the model produce a mask and calculate meanIoU for each mask
def run_box(efficientvit_sam, dataloader, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()                 # move model to correct GPU, and turn on eval mode
    predictor = EfficientViTSamPredictor(efficientvit_sam)                      # create predictor

    output = []
    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):        # for each batch of images
        if i == 25:
            break
        data = data[0]                                                          # fetch the images?
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))     # convert ot RGB image
        predictor.set_image(sam_image)                                          # send image to predictor
        anns = data["anns"]                                                     # fetch annotations for the batch
        for ann in anns:                                                        # for each annotation
            if ann["area"] < 1:                                                 # skip of too small bounding box
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


def calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank):
    efficientvit_sam = efficientvit_sam.cuda(local_rank).eval()                 # move model to correct GPU, and turn on eval mode
    predictor = EfficientViTSamPredictor(efficientvit_sam)                      # create predictor
  
    efficientvit_sam.toggle_calibrate_on()                                      # sets calibrate = true for all 'relevant' modules

    for i, data in enumerate(tqdm(calib_dataloader, disable=local_rank != 0)):  # for each batch of images
        if i == len(calib_dataloader) - 1 or i == args.limit_iterations - 1:                                       # The lenght of the dataloader is dynamicas data is split over GPUs. zero-based enumeration
            print("Did reach last_calibration, with i = ", i, len(calib_dataloader))
            efficientvit_sam.toggle_last_calibrate_on()                               # if second to last batch, set last_calibrate = true for all relevant modules
        if i == args.limit_iterations:
            break
        data = data[0]                                                          # fetch the images?
        sam_image = np.array(Image.open(data["image_path"]).convert("RGB"))     # convert ot RGB image
        predictor.set_image(sam_image)                                          # send image to predictor
        anns = data["anns"]                                                     # fetch annotations for the batch
        for ann in anns:                                                        # for each annotation
            if ann["area"] < 1:                                                 # skip of too small bounding box
                continue
            bbox = np.array(bbox_xywh_to_xyxy(ann["bbox"]))                     # find bounding box - (i.e. the prompt)
            _ = predict_mask_from_box(predictor, bbox)                          # predict mask from bounding box
    
    efficientvit_sam.toggle_calibrate_off()                                 # sets calibrate = false for all reelvant modules
    efficientvit_sam.toggle_last_calibrate_off()                            # sets last_calibrate = false for all reelvant modules


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


def calibrate_run_point(efficientvit_sam, calib_dataloader, args, local_rank):
    raise NotImplementedError()


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


def calibrate_run_box_from_detector(efficientvit_sam, calib_dataloader, args, local_rank):
    raise NotImplementedError()


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

def evaluate_to_dataframe(dataframe, results, prompt_type, dataset, annotation_json_file=None, args=None):
    # append each new result to the dataframe, and return the dataframe
    if prompt_type == "point" or prompt_type == "box":
        metrics = get_iou_metric(results)
        for key, val in metrics.items():
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

def create_dataframe(prompt_type, columns, script_name: str) -> pd.DataFrame:
    # add columns
    if prompt_type == 'box':
            columns.extend([
            "all",
            "large",
            "medium",
            "small", 
            ])
    elif prompt_type == "point":
        raise NotImplementedError()
    elif prompt_type == "box_from_detector":
        raise NotImplementedError()
    else:
        raise NotImplementedError()
    
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

def metadata_to_dataframe(dataframe: pd.DataFrame, args, columns) -> pd.DataFrame:
    row_data = {}
    for column in columns:
        row_data[column] = getattr(args, column)
    dataframe = pd.concat([dataframe, pd.DataFrame([row_data])], ignore_index=True)
    return dataframe

def save_dataframe_to_file(dataframe: pd.DataFrame, script_name: str) -> None:
    # Save the dataframe as a pickle file in the 'results' directory. Will overwrite.
    # remove any leading directories and extensions from the script name
    script_name = os.path.basename(script_name)
    script_name = os.path.splitext(script_name)[0]
    dataframe.to_pickle(path=f'results/{script_name}.pkl')

'''
        # Cosntant: Weight quant = only in matmul.
        Variables:
        First input conv: always no.
        Attention stages as a whole yes-no
            Attention Relu sub-block yes or no
            Attention scaling convs yes or no
            Attention final MBConv (which is not a solo bottleneck) yes or no
        Convolutional stages yes or no
        Bottlenecks yes or no
            Bottlenecks in attention-stages yes or no
            Bottlenecks in conv-stages yes or no
        SAM-Neck yes or no

        Calibration method?
        Layer-wise vs channel-wise?
        Linear matmul in the attention layers yes or no
    '''
''' Attempt 1: "Save attention-stages, Q the bulk of the buildup"
        - First Layer: No
        - Stage 1,2,3: Yes, Q the Convs
        - Bottlenecks: No, don't Q them ever, they could be important
        - Attention, stage 4 (+5): No, save them.
        - Neck: No
        - Linear matmul in the attention layers: no'''
''' Attempt 2: "Q the conv-parts in the attention, leave the rest"
        - First Layer: No
        - Stage 1,2,3: No
        - Bottlenecks: No, don't Q them ever, they could be important
        - Attention, stage 4 (+5): Yes, Q the convolutions in both scaling, QKV, and projection, and output
        - Neck: No
        - Linear matmul in the attention layers: no'''

def quantize(efficientvit_sam, backbone_version='0', suppress_print=False):
    printout=(local_rank==0 and suppress_print is False)

    if backbone_version == '0':
        ''' Baseline backbone:
        Quant was applied to all Conv-blocks exlusive of EfficientVi-Modules and the input layer. Neck was also spared.'''
        efficientvit_sam.toggle_quant_on() # just sets module.quant = true (or = 'int'). Doesn't alter any weights!
    
    elif backbone_version == '1a':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage0", "stage1", "stage2", "stage3"],
            block_names = ["res", "fmb", "fmb", "mb"],
            spare_bottlenecks=True,
            printout=printout,
            )
    
    elif backbone_version == '1b':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage0"],
            block_names = ["res", "fmb", "fmb", "mb"],
            spare_bottlenecks=True,
            printout=printout,
            )
        
    elif backbone_version == '1c':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage1"],
            block_names = ["res", "fmb", "fmb", "mb"],
            spare_bottlenecks=True,
            printout=printout,
            )
    
    elif backbone_version == '1d':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage2"],
            block_names = ["res", "fmb", "fmb", "mb"],
            spare_bottlenecks=True,
            printout=printout,
            )
    
    elif backbone_version == '1e':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage3"],
            block_names = ["res", "fmb", "fmb", "mb"],
            spare_bottlenecks=True,
            printout=printout,
            )

    elif backbone_version == '2a':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage4", "stage5"],
            block_names = ["att", "att@3", "att@5"],
            spare_bottlenecks=True,
            spare_attention_qkv=False,
            spare_attention_scaling=False,
            spare_attention_projection=False,
            printout=printout,
            )
        
    elif backbone_version == '2b':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage4", "stage5"],
            block_names = ["att", "att@3", "att@5"],
            spare_bottlenecks=True,
            spare_attention_qkv=True,
            spare_attention_scaling=False,
            spare_attention_projection=False,
            printout=printout,
            )

    elif backbone_version == '2c':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage4", "stage5"],
            block_names = ["att", "att@3", "att@5"],
            spare_bottlenecks=True,
            spare_attention_qkv=False,
            spare_attention_scaling=True,
            spare_attention_projection=False,
            printout=printout,
            )

    elif backbone_version == '2d':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage4", "stage5"],
            block_names = ["att", "att@3", "att@5"],
            spare_bottlenecks=True,
            spare_attention_qkv=False,
            spare_attention_scaling=False,
            spare_attention_projection=True,
            printout=printout,
            )
        
    elif backbone_version == '2e':
        efficientvit_sam.toggle_selective_quant_on(
            stages = ["stage4", "stage5"],
            block_names = ["att", "att@3", "att@5"],
            spare_bottlenecks=True,
            spare_attention_qkv=True,
            spare_attention_scaling=True,
            spare_attention_projection=True,
            printout=printout,
            )

    else:
        print("Version not yet implemented")
        raise NotImplementedError("Backbone version not yet implemented")














if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)
    parser.add_argument("--prompt_type", type=str, default="point", choices=["point", "box", "box_from_detector"])
    parser.add_argument("--num_click", type=int, default=1)
    parser.add_argument("--dataset", type=str, choices=["coco", "lvis"])
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--image_root_calibration", type=str, default="coco/minitrain2017")
    parser.add_argument("--annotation_json_file", type=str)
    parser.add_argument("--source_json_file", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--single_gpu', action='store_true', help="Force the use of a single gpu, might help in troubleshooting quantization")
    parser.add_argument('--suppress_print', action="store_true", help="suppresses debugging printouts")
    parser.add_argument('--export_dataframe', action="store_true")
    parser.add_argument('--script_name', type=str)
    parser.add_argument('--limit_iterations', type=int, default=-1)

    parser.add_argument("--quantize", action="store_true", help="Turn on quantization and calibration for weights, activations, or norms")
    parser.add_argument("--quantize_W", action="store_true", help="Turn on quantization and calibration for weights")
    parser.add_argument("--quantize_A", action="store_true", help="Turn on quantization and calibration for activations")
    parser.add_argument("--quantize_N", action="store_true", help="Turn on quantization and calibration for norms")

    parser.add_argument("--observer_method_W", default="minmax", choices=["minmax", "ema", "omse", "percentile"]) #TODO - implement this
    parser.add_argument("--observer_method_A", default="minmax", choices=["minmax", "ema", "omse", "percentile"]) #TODO - implement this
    parser.add_argument("--observer_method_N", default="minmax", choices=["minmax", "ema", "omse", "percentile"]) #TODO - implement this

    parser.add_argument("--quantize_method_W", default="uniform", choices=["uniform", "log2"]) #TODO - implement this
    parser.add_argument("--quantize_method_A", default="uniform", choices=["uniform", "log2"]) #TODO - implement this
    parser.add_argument("--quantize_method_N", default="uniform", choices=["uniform", "log2"]) #TODO - implement this

    parser.add_argument("--backbone_version", type=str, default='0')


    args = parser.parse_args()
    # Set args.quantize to True if any of the other quantize arguments are True
    args.quantize = args.quantize_W or args.quantize_A or args.quantize_N

    # TODO: Implement for all three val types
    # TODO: Quantize norms and activations, i.e. make the config work.
    
    # colums for dataframes when running scripts.
    # TODO: Perhaps these can be built from the arguments of the Config object instead? Not all but half.
    columns = [
        "model",
        "prompt_type",
        "backbone_version",
        "quantize_W",
        "quantize_A",
        "quantize_N",
        "observer_method_W",
        "observer_method_A",
        "observer_method_N",
        "quantize_method_W",
        "quantize_method_A",
        "quantize_method_N",
        "num_click",
        "dataset",
        "image_root",
        "image_root_calibration",
        "limit_iterations"
    ]

    config = Config(args) # quantization configuration
    if args.single_gpu:
        local_rank = 0
        if local_rank == 0 and not args.suppress_print: # only master process prints
            print("Using single GPU")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl") # initializing the distributed environment 
        if local_rank == 0 and not args.suppress_print:
            print(f"Using {torch.distributed.get_world_size()} GPUs")
    torch.cuda.set_device(local_rank)
    
    # model creation
    efficientvit_sam = create_sam_model(name=args.model, pretrained=True, weight_url=args.weight_url, config=config)

    # dataset creation
    dataset = eval_dataset(
        args.dataset, args.image_root, args.prompt_type, args.annotation_json_file, args.source_json_file
    )
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
    )
    if local_rank == 0 and not args.suppress_print:
        print(f"The dataloader contains {len(dataloader.dataset)} images.")

    # calibration dataset
    if args.quantize:
        calib_dataset = eval_dataset(
            args.dataset, args.image_root_calibration, args.prompt_type, args.annotation_json_file, args.source_json_file
        )
        calib_sampler = DistributedSampler(calib_dataset, shuffle=False)
        calib_dataloader = DataLoader(
            calib_dataset, batch_size=1, sampler=calib_sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
        )
        if local_rank == 0 and not args.suppress_print:
            print(f"The calibration dataloader contains {len(calib_dataloader.dataset)} images.")

    # inference + calibration + quantization
    if args.prompt_type == "point":
        if args.quantize:
            if local_rank == 0 and not args.suppress_print:
                print("Calibrating point...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam, args.backbone_version, args.suppress_print)

        results = run_point(efficientvit_sam, dataloader, args.num_click, local_rank)

    elif args.prompt_type == "box":
        if args.quantize:
            if local_rank == 0 and not args.suppress_print:
                print("Calibrating box...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam, args.backbone_version, args.suppress_print)

        results = run_box(efficientvit_sam, dataloader, local_rank)

    elif args.prompt_type == "box_from_detector":
        if args.quantize:
            if local_rank == 0 and not args.suppress_print:
                print("Calibrating box_from_detector...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam, args.backbone_version, args.suppress_print)

        results = run_box_from_detector(efficientvit_sam, dataloader, local_rank)

    else:
        raise NotImplementedError()

    # evaluation - only done my the master process, not other parallell processes
    if local_rank == 0:
        if args.export_dataframe:
            df = create_dataframe(args.prompt_type, columns.copy(), args.script_name)
            df = metadata_to_dataframe(df, args, columns)
            df = evaluate_to_dataframe(df, results, args.prompt_type, args.dataset, args.annotation_json_file, args=args)
            print("New row added to results: \n", df.tail(1))
            save_dataframe_to_file(df, args.script_name)
        else:
            evaluate(results, args.prompt_type, args.dataset, args.annotation_json_file)
