# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import argparse
import math
import os

import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from efficientvit.apps.utils import AverageMeter
from efficientvit.cls_model_zoo import create_cls_model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list[torch.Tensor]:
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, 1, True, True)                              # highest maxk values and their indicies
    pred = pred.t()                                                         # transpose
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))                # softmax is not needed because accuracy is only concerened with which indicies hold the largest entries

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
    #parser.add_argument("--model", type=str)
    # below: model hradcoded to skip arguments when developing
    parser.add_argument("--model", type=str, default="b1_quant-r224")
    parser.add_argument("--weight_url", type=str, default=None)
    # quantization arguments
    parser.add_argument("--quantize", action="store_true") # just flag --quantize to turn on quantization
    parser.add_argument('--calib-batchsize', default=50,type=int,help='batchsize of calibration set') # FQ-ViT used 100, not 50
    parser.add_argument('--calib-iter', default=10, type=int)

    args = parser.parse_args()

    # GPU connection
    if args.gpu == "all":
        device_list = range(torch.cuda.device_count())
        args.gpu = ",".join(str(_) for _ in device_list)
    else:
        device_list = [int(_) for _ in args.gpu.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.batch_size = args.batch_size * max(len(device_list), 1) # multiply by numer of GPUs to benefit from parallell processing

    # Transformations
    val_transform = transforms.Compose([
                        transforms.Resize(int(math.ceil(args.image_size / args.crop_ratio)), interpolation=InterpolationMode.BICUBIC),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    # Datasets
    val_dataset = datasets.ImageFolder(args.path, val_transform)

    # Dataset loaders
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.quantize: calib_data_loader = val_data_loader

    # Model creation
    #  model is created and b1_quant is passed as args.model
    # "model" is an instance of class EfficientVitCls, incl. weights
    model = create_cls_model(args.model, weight_url=args.weight_url)
    model = torch.nn.DataParallel(model).cuda() # wrapper -> To reach the model's own methods, call model.module.method
    model.eval()

    # Calibration dataset creation
    if args.quantize:
        # Using validation data now - must change to training data!
        calibration_dataset = []
        for i, (images, labels) in enumerate(calib_data_loader):
            if i == args.calib_iter: break          # default is 10 iterations
            immages = images.cuda()                 # move data to GPU
            calibration_dataset.append(images)      # append each batch

        print("Calibrating...")
        model.module.toggle_calibrate_on()                # sets calibrate = true for all 'relevant' modules
        with torch.no_grad():
            for i, images in enumerate(calibration_dataset):
                if i == len(calibration_dataset) - 1:           # for each batch of calibration data
                    model.module.toggle_last_calibrate_on()            # if last batch, set last_calibrate = true for all relevant modules
                _ = model(images)                               # feed forward
            model.module.toggle_calibrate_off()                        # sets calibrate = false for all reelvant modules
            model.module.toggle_last_calibrate_off()                   # sets last_calibrate = false for all reelvant modules
            model.module.model_quant()                                 # just sets module.quant = true (or = 'int'). Doesn't alter any weights!

    # Evaluation
    print("Validating...")
    validate(args, val_data_loader, model)

    print_model_params(model)
    
    #print(f"Model structure: {model}\n\n")

def validate(args, val_data_loader, model):
    top1 = AverageMeter(is_distributed=False)
    top5 = AverageMeter(is_distributed=False)
    with torch.inference_mode():
        with tqdm(total=len(val_data_loader), desc=f"Eval {args.model} on ImageNet") as t:
            for images, labels in val_data_loader:                      # images is a 4D tensor: (batch_size, channels (rgb), height, width)
                images, labels = images.cuda(), labels.cuda()       # Moving images and labels to the GPU
                output = model(images)                              # inference
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))  # measure accuracy and record loss

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

    print("Results from Joel's evaluation")
    print(f"Top1 Acc={top1.avg:.3f}, Top5 Acc={top5.avg:.3f}")

# prints the first 10 model params
def print_model_params(model):
    count = 0
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} |\n")
        count = count +1
        if count > 10:
            break

if __name__ == "__main__":
    main()
