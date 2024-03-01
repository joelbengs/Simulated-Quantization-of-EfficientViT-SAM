# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import math
import os
import time
import numpy as np

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm #library that prints a progress bar when iterating

from efficientvit.apps.utils import AverageMeter # an interface for calculating an average simply.
from efficientvit.cls_model_zoo import create_cls_model # an interface for creating the EfficientViT models

# import common # for common.allocate_buffers(engine)
import tensorrt as trt # TENSORRT - but what version?
import onnx # model format
import onnx_tensorrt.backend as backend # ONNX + TensorRT backend, might not be what we want?

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/dataset/imagenet/val")
    parser.add_argument("--gpu", type=str, default="all")
    parser.add_argument("--batch_size", help="batch size per gpu", type=int, default=50)
    parser.add_argument("-j", "--workers", help="number of workers", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_ratio", type=float, default=0.95)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_url", type=str, default=None)

    # configure GPUs
    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1)

    # load ImageNet dataset
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            args.path,
            transforms.Compose(
                [
                    transforms.Resize(
                        int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    # create normal classification model using PyTorch
    model = create_cls_model(args.model, weight_url=args.weight_url)
    
    # ONNX EXPORT from PyTorch, no TRT involved.
    # We need a batch of data to save our ONNX file from PyTorch. We will use a dummy batch.
    input_shape = (1, 3, 224, 224)
    input_names = ['input']
    output_names = ['output']
    dummy_input = torch.randn(input_shape)
    torch.onnx.export(model, dummy_input, 'my_model.onnx', verbose=False, input_names=input_names, output_names=output_names)
    
    # Evaluate model with pytorch
    model = torch.nn.DataParallel(model).cuda()
    model.eval() # Sets the module in evaluation mode, does not do evaluation
    top1 = AverageMeter(is_distributed=False) # score counter
    top5 = AverageMeter(is_distributed=False) # score counter
    with torch.inference_mode():
        with tqdm(total=len(data_loader), desc=f"Eval {args.model} on ImageNet") as t:
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                # compute output
                output = model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "resolution": images.shape[-1],
                    }
                )
                t.update(1)
    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")

    # IMPORT ONNX
    # Adopted from 4. The Python API in the documentation of TRT
    # example here: https://github.com/NVIDIA/TensorRT/blob/main/samples/python/introductory_parser_samples/onnx_resnet50.py
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) #The EXPLICIT_BATCH flag is required in order to import models using the ONNX parser
    parser = trt.OnnxParser(network, logger)
    # parse the onnx model using the parser
    success = parser.parse_from_file('my_model.onnx')
    # process errors, if any
    if not success:
        print("Failed to parse ONNX model.")
        for idx in range(parser.num_errors):
            print(parser.get_error(idx))

    # BUILDER, ENGINE
    # Building an Engine and serializing "the plan"
    config = builder.create_builder_config()
    print("TensorRT building optimized model...")
    serialized_engine = builder.build_serialized_network(network, config)
    # can be saved to file here for future use.
    print("serialized engine completed")

    # Deserializing a plan to perform inference using 
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine) # return type: ICudaEngine
    # could be loaded from a file instead of from memory
    print("deserialization complete")
    # CONTEXT; INFERENCE
    # Context needed for intermediate activations in inference.
    # Allocate buffers and create a CUDA stream.
        #inputs, outputs, bindings, stream = trt.common.allocate_buffers(engine)
        #context = engine.create_execution_context()

        #trt_outputs = trt.common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    
    # BELOW IS COPY FRMO ARTICLE ON MEDIUM: https://medium.com/@zergtant/accelerating-model-inference-with-tensorrt-tips-and-best-practices-for-pytorch-users-7cd4c30c97bc
    # Allocate device memory for input and output buffers
    input_name = 'input'
    output_name = 'output'
    input_shape = (1, 3, 224, 224) # from Puren's
    output_shape = (1, 5) # just copied
    input_buf = trt.cuda.alloc_buffer(builder.max_batch_size * trt.volume(input_shape) * trt.float32.itemsize)
    output_buf = trt.cuda.alloc_buffer(builder.max_batch_size * trt.volume(output_shape) * trt.float32.itemsize)

    # Create a TensorRT execution context
    context = engine.create_execution_context()

    # Run inference on the TensorRT engine
    input_data = torch.randn(1, 3, 224, 224).numpy()
    output_data = np.empty(output_shape, dtype=np.float32)
    input_buf.host = input_data.ravel()
    trt_outputs = [output_buf.device]
    trt_inputs = [input_buf.device]
    context.execute_async_v2(bindings=trt_inputs + trt_outputs, stream_handle=trt.cuda.Stream())
    output_buf.device_to_host()
    output_data[:] = np.reshape(output_buf.host, output_shape)

    # Print the output
    print(output_data)
    
if __name__ == "__main__":
    main()
