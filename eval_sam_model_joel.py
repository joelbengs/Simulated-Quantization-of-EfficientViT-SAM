
# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import json
import os

import numpy as np
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
  
    efficientvit_sam.toggle_calibrate_on()                               # sets calibrate = true for all 'relevant' modules

    for i, data in enumerate(tqdm(dataloader, disable=local_rank != 0)):        # for each batch of images
        if i == args.calib_iter:
            break                                          # default is 10 batches
        elif i == args.calib_iter - 1:
            efficientvit_sam.toggle_last_calibrate_on()        # if last batch, set last_calibrate = true for all relevant modules
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

def quantize(efficientvit_sam):
    efficientvit_sam.toggle_quant_on()                             # just sets module.quant = true (or = 'int'). Doesn't alter any weights!


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
    parser.add_argument("--quantize", action="store_true", help="Turn on quantization and calibration")
    parser.add_argument("--calib_iter", type=int, default=100)
    parser.add_argument("--quant-method-W", default="minmax", choices=["minmax", "ema", "omse", "percentile"])
    parser.add_argument("--quant-method-A", default="minmax", choices=["minmax", "ema", "omse", "percentile"]) #TODO - implement this
    parser.add_argument('--single_gpu', action='store_true', help="Force the use of a single gpu, might help in troubleshooting quantization")

    args = parser.parse_args()
    
    # TODO: implement different calibration types
    # TODO: Implement for all three val types
    # TODO: Get coco traindata for calibration
    # TODO: Start building different backbones for quant of different parts
    # TODO: Quantize norms and activations, i.e. make the config work.


    config = Config(args.quant_method_W, args.quant_method_A) # quantization configuration

    if args.single_gpu:
        local_rank = 0
        print("Using single GPU")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl") # initializing the distributed environment 
        if local_rank == 0: # only master process prints
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

    # calibration dataset
    if args.quantize:
        print("Loading minitrain dataset")
        calib_dataset = eval_dataset(
            args.dataset, args.image_root_calibration. args.prompt_type, args.__annotation_json_file, args.source_json_file
        )
        calib_sampler = DistributedSampler(calib_dataset, shuffle=False)
        calib_dataloader = DataLoader(
            calib_dataset, batch_size=1, sampler=sampler, drop_last=False, num_workers=args.num_workers, collate_fn=collate_fn
        )

    #calib_dataloader = dataloader # Using validation data now - must change to training data!

    # inference + calibration + quantization
    if args.prompt_type == "point":
        if args.quantize:
            print("Calibrating point...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam)

        results = run_point(efficientvit_sam, dataloader, args.num_click, local_rank)

    elif args.prompt_type == "box":
        if args.quantize:
            print("Calibrating box...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam)

        results = run_box(efficientvit_sam, dataloader, local_rank)

    elif args.prompt_type == "box_from_detector":
        if args.quantize:
            print("Calibrating box_from_detector...")
            calibrate_run_box(efficientvit_sam, calib_dataloader, args, local_rank)
            quantize(efficientvit_sam)

        results = run_box_from_detector(efficientvit_sam, dataloader, local_rank)

    else:
        raise NotImplementedError()

    # evaluation - only done my the master process, not other parallell processes
    if local_rank == 0:
        evaluate(results, args.prompt_type, args.dataset, args.annotation_json_file)
