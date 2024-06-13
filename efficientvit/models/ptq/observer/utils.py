# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented into EfficientViT-SAM for quantization simulation
def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()
