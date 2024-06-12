# Simulated Quantization of EfficientViT-SAM

## Datasets

Evaluation on [COCO2017](https://cocodataset.org/#download) and [LVIS annotations](https://www.lvisdataset.org/dataset) is supported.

To conduct box-prompted instance segmentation with detected boxes, you must first obtain the *source_json_file* of those boxes. Follow the instructions of [ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet), to get the *source_json_file*. You can also download our [pre-generated files](https://huggingface.co/han-cai/efficientvit-sam/tree/main/source_json_file). Our thesis onducted box-prompted instance ssegmentation of the COCO ground truth (measuring mIoU) and the ViTDET boxes (measuring mAP). Below is the expected file structure. Note that train2017 is only needed only for LVIS evaluation, not COCO evaluation. The model is trained and calibrated on SA-1B, so evaluation on COCO train2017 is okay. Resonably, place the entire coco/ directory on a shared server, mount it as a volume to the container, and symlink to it from the container.

<details>
<summary>Expected directory structure:</summary>

```python
coco
├── train2017
├── val2017
├── annotations
│   ├── instances_val2017.json
│   ├── lvis_v1_val.json
|── source_json_file
│   ├── coco_vitdet.json
│   ├── lvis_vitdet.json
```
</details>

## Quantized models

__LEFT__: Accuracy vs size trade-off. Accuracy is measured in simulation, the size of the model refers to the size of the weights only, under the given mixed-precision scheme (FP16/INT8). Only L-series included, since accuracy drops severly for the XL-Series.

__RIGHT__: Accuracy vs latency trade-off. Here, accuracy is measured in deployment using TensorRT. This highlights that __the simulation framework fails to approximate the accuracy results in deployment!__.

<div style="display: flex; justify-content: space-around;">
    <div>
        <p align="left">
            <img src="plots/graphs/size_green_removed_zoomed.png"  width="450">
        </p>
    </div>
    <div>
        <p align='left'>
            <img src="plots/graphs/latency_green_removed.png" width="450">
        </p>
    </div>
</div>

## Pretrained non-quantized models, as compared to the original SAM

Latency/Throughput is measured on NVIDIA Jetson AGX Orin, and NVIDIA A100 GPU with TensorRT, fp16. Data transfer time is included.

<p align="left">
<img src="assets/files/sam_zero_shot_coco_mAP.png"  width="450">
</p>


## Usage

### main workflow
```python
# file: eval_sam_quant_model.py
# model creation 
from efficientvit.sam_model_zoo import create_sam_model
from quant_config import Config

quant_config = Config(args)
efficientvit_sam = create_sam_model(
    name="l0_quant", pretrained=True, weight_url="assets/checkpoints/sam/l0.pt", config=quant_config
    )

efficientvit_sam = efficientvit_sam.cuda().eval()

# calibration of quantization operators
calib_dataloader = DataLoader(calib_dataset, ...)
calibrate_image_encoder(efficientvit_sam, calib_dataloader, args, local_rank)

# activate quantization
toggle_operation(efficientvit_sam, 'quant_weights', 'on')
toggle_operation(efficientvit_sam, 'quant_activations', 'on')

# inference and evaluation of quantized model
dataloader = DataLoader(dataset, ...)
results = run_box(efficientvit_sam, dataloader)
evaluate(results, ...)
```

### script: layer-wise sensitivity analysis

```python
# launch layer-wise sensitvity analysis of all model variants
scripts/layer_analysis_script.sh

# expected output:
{--------- Layer-wise sensitivity analysis of EfficientViT-SAM L0 ---------}
Model l0_quant, backbone_version: L0:stage0:0:0 §
 45%|████████        | 2250/5000 [03:22<04:07, 11.10it/s]
100%|████████████████| 2500/2500 [01:56<00:00, 21.42it/s]
saved 0.00 Mb
New row added to file results/l0_quant.pkl: 
    model backbone_version prompt_type quantize_W quantize_A ... megabytes_saved
0   l0_quant    L0:stage0:0:0         box       True       True  ...       0.000824

Model l0_quant, backbone_version: L0:stage0:1:0 §
...
[same pattern for all layers of all model variants.]
```
### script: Mix-DWSC evaluation

```python
# evaluate quantization scheme Mix-DWSC
# Step 1: Manually enter efficientvit/models/nn/ops.py/class QMBConv and configure:
#        protect_sensitive_input_conv_to_FP32 = False
#        protect_sensitive_depthwise_conv_to_FP32 = True
#        protect_sensitive_pointwise_conv_to_FP32 = True
# Step 2: run script
scripts/benchmark_mix_dwsc.sh

# expected output: 
--------- Benchmarking Mix-DWSC on COCO (remeber to toggle two flags in OPS.py) ---------
 
 90%|███████████████| 4500/5000 [06:52<00:45, 10.91it/s]
100%|█████████████████| 2500/2500 [02:40<00:00, 15.57it/s]

all=75.757, large=80.210, medium=77.989, small=71.337
saved 18.55 Mb
box COCO: l0_quant, any:all:all:all §
...
[same pattern for the five model variants, followed by evaluation of LVIS, followed by evaluation of detected box-prompts on both COCO and LVS]
```

### script: Mix-MBC-Neck evaluation

```python
# evaluate quantization scheme Mix-MBC-Neck
# Step 1: Manually enter efficientvit/models/nn/ops.py/class QMBConv and configure:
#        protect_sensitive_input_conv_to_FP32 = True
#        protect_sensitive_depthwise_conv_to_FP32 = True
#        protect_sensitive_pointwise_conv_to_FP32 = True
# Step 2: run script
scripts/benchmark_mix_mbc-neck.sh

# expected output: 
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=300 catIds=all] = 0.444

```

### script: Full INT8 evaluation

```python
# evaluate quantization to INT8 of the entire encoder:
# Step 1: make sure that the toggles in efficientvit/models/nn/ops.py/class QMBConv are all False
# Step 2: run script (which is equivalent to the script for mix-dwsc)
scripts/benchmark_int8.sh

```

### Design your own quantization schemes

There are two main ways to design a quantization scheme.

First, one can enter `ops.py` and manually substitute the quantized layers (`QConvLayer`) to non-quantized versions (`ConvLayer`). One step higher, one can enter `backbone.py` and edit the class `EfficientViTLargeBackboneQuant` by substituting quantized blocks (such as QMBConv) for non-quantized versions (MBConv). Then just benchmark a model using scripts/benchmark_int8.sh.

Second, one can design a custom backbone version in `quant_backbone_zoo.py`, and then use that as `--backbone_version` argument to the main method. The logic is as follows. A dictinary is specified on this format:

```python
# example scheme that applies quantization to point-wise convs in Stage 1 and 2, nothing else.
    backbone_dict['L0:1,2:any:1'] =  {
        'stages': [1,2],
        'block_positions': [],
        'layer_positions': [1],
    }
```
This backbone version will cause _only_ those layers that match all three criterions to be quantized. An empty [] means that _all_ layers will automatically pass this criterion. In this example, the quantization will target layers in stage 1 and 2, at any block position withing the stages. It will hoever only target layer-position 1 in each block. This is equivalent to quantization of the 1x1 point-wise convolutions inside each Fused-MBConv block in those stages. 'L0' here means model L0, but substitute for 'any' to make the same scheme applicable to any model. The scheme will be forwarded to the function `toggle_selective_attribute` in `class EfficientViTSamImageEncoder` in `sam.py`

One can use the dictionaries `REGISTERED_BACKBONE_DESCRIPTIONS` (L/XL) to find out what layer has what position, on the format stage:block:layer.  Indexing is always zero-based.

## Code files important for quantizatiion

