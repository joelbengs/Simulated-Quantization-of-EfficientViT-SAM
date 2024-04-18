
10 samples calibration -> 320mb tensor in 1:0:0 as maximum
size of act tensors
stage0:0:0:act:independent 2.10e+07
stage0:1:0:act:res 2.10e+07
stage1:0:0:act:fmb 8.39e+07 # largest
stage1:1:0:act:fmb 4.19e+07
stage2:0:0:act:fmb 4.19e+07
stage2:1:0:act:fmb 2.10e+07
stage3:0:0:act:mb 8.39e+07
stage3:0:1:act:mb 2.10e+07
stage3:1:0:act:mb 1.05e+07
stage3:1:1:act:mb 1.05e+07
stage3:2:0:act:mb 1.05e+07
stage3:2:1:act:mb 1.05e+07
stage3:3:0:act:mb 1.05e+07
stage3:3:1:act:mb 1.05e+07
stage3:4:0:act:mb 1.05e+07
stage3:4:1:act:mb 1.05e+07
stage4:0:0:act:mb 6.29e+07
stage4:0:1:act:mb 1.57e+07
stage4:1:4:act:att 7.86e+06
stage4:1:5:act:att 7.86e+06
stage4:2:4:act:att 7.86e+06
stage4:2:5:act:att 7.86e+06
stage4:3:4:act:att 7.86e+06
stage4:3:5:act:att 7.86e+06
stage4:4:4:act:att 7.86e+06
stage4:4:5:act:att 7.86e+06
neck:3:0:act:fmb 1.05e+07
neck:4:0:act:fmb 1.05e+07
neck:5:0:act:fmb 1.05e+07
neck:6:0:act:fmb 1.05e+07
  0%|▎                          


100 sample calibration, took 3 minutes -> 3.2gb tensor in 1:0:0
  size of act tensors
stage0:0:0:act:independent 2.10e+08
stage0:1:0:act:res 2.10e+08
stage1:0:0:act:fmb 8.39e+08 #largest
stage1:1:0:act:fmb 4.19e+08
stage2:0:0:act:fmb 4.19e+08
stage2:1:0:act:fmb 2.10e+08
stage3:0:0:act:mb 8.39e+08
stage3:0:1:act:mb 2.10e+08
stage3:1:0:act:mb 1.05e+08
stage3:1:1:act:mb 1.05e+08
stage3:2:0:act:mb 1.05e+08
stage3:2:1:act:mb 1.05e+08
stage3:3:0:act:mb 1.05e+08
stage3:3:1:act:mb 1.05e+08
stage3:4:0:act:mb 1.05e+08
stage3:4:1:act:mb 1.05e+08
stage4:0:0:act:mb 6.29e+08
stage4:0:1:act:mb 1.57e+08
stage4:1:4:act:att 7.86e+07
stage4:1:5:act:att 7.86e+07
stage4:2:4:act:att 7.86e+07
stage4:2:5:act:att 7.86e+07
stage4:3:4:act:att 7.86e+07
stage4:3:5:act:att 7.86e+07
stage4:4:4:act:att 7.86e+07
stage4:4:5:act:att 7.86e+07
neck:3:0:act:fmb 1.05e+08
neck:4:0:act:fmb 1.05e+08
neck:5:0:act:fmb 1.05e+08
neck:6:0:act:fmb 1.05e+08


So some more results to share.



Here is layerwise analysis. One layer each time. A layer is defined as conv + norm + activation, if norm/activation exists. Red line is baseline. This is only model L0. Y-axis is a performance measure, think accuracy.



The notation on the X-axis is "stage number : block number : layer number". All numbers are zero based.



The interesting dips actually stem from only three layers:



FusedMBConvs exit layer: All dips in stage 0 /1/2 come from the last layer of the FMBConv. This layer has no activation function, it is just conv + batchnorm.



MBConv exit layer: All dips in stage 3 show the same behaviour: the last layer of each MBConv module. The difference between MBConv and Fused MBConv is just that the first two (of total three) layers have been fused. The exit layer is the same in both. It has no activation, again.



Attention scaling pointwise layer: The dips inside the attention blocks all come from layer 4:x:2, where x = 1,2,3,4. These are pointwise convolutions inside the scaling. Simple convs with kernel size = 1. No activations, no norms!



There are some dips in stage4 stemming from MB-conv exit layers as well.

