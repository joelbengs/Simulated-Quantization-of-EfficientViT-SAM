# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val.to(v.device)) # moving to same cuda, FQ-ViT was implemented on single GPU
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val.to(v.device)) # moving to same cuda, FQ-ViT was implemented on single GPU

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()
        
        # to ensure self.max_val and self.min_val are on the same device as self.device
        self.max_val = self.max_val.to(self.device)
        self.min_val = self.min_val.to(self.device)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val.to(self.device) # moving to same cuda
        min_val = self.min_val.to(self.device) # moving to same cuda

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64) # Why int64?

        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)

        # Error checking
        if scale is None or zero_point is None:
            raise ValueError("Scale or zero_point is None. They should be valid tensors.")

        return scale, zero_point
