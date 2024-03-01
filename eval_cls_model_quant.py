# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import math
import os
import time

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model

import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend

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

    args = parser.parse_args()
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.batch_size = args.batch_size * max(len(device_list), 1)

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

    model = create_cls_model(args.model, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()   
    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
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



    # TENSORRT QUANTIZATION WORK
    print("Quantization process starting...")
    
    # Create a new classification model
    model = create_cls_model(args.model, weight_url=args.weight_url)

    input_shape = (1, 3, 224, 224)
    input_names = ['input']
    output_names = ['output']

    # an input sample of just random numbers
    dummy_input = torch.randn(input_shape)

    #export an onnx model, no TRT involved.
    torch.onnx.export(model, dummy_input, 'my_model.onnx', verbose=False, input_names=input_names, output_names=output_names)
    
    #omport an onnx model, no TRT involved.
    model_onnx = onnx.load('my_model.onnx')

    # create a tensorRT engine using onnx_tensorrt.backend
    engine = backend.prepare(model_onnx, device='CUDA:0')

    #context = engine.create_execution_context()

    model.eval()

    trt_engine = backend.prepare(model_onnx, device='CUDA:0')
    num_iterations = 2
    total_time = 0.0
    with torch.no_grad():
        for i in range(num_iterations):
            input_data = torch.randn(input_shape).cuda()
            start_time = time.time()
            output_data = trt_engine.run(input_data.cpu().numpy())[0]
            end_time = time.time()
            total_time = total_time + (end_time - start_time)

    tensorrt_fps = num_iterations/total_time
    print(f"TensorRT FPS: {tensorrt_fps:.2f}")

    
if __name__ == "__main__":
    main()
