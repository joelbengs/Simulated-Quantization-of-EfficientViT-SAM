# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn

from .base import BaseQuantizer


class UniformQuantizer(BaseQuantizer):

    def __init__(self, bit_type, observer, module_type):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type)
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)
        if self.scale is None:
            raise ValueError("Scale is None after getting quantization params.")
        if self.zero_point is None:
            raise ValueError("Zero point is None after getting quantization params.")
        #print(f"Parameters updated: S = {self.scale}, S.type = {self.scale.type()}, Z = {self.zero_point}, Z.type = {self.zero_point.type()}")


    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        outputs = outputs.round().clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    # Note: you can also dequantize as easy as quant(input, scale, zeropoint).
    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
    
    # forward() is definined in class BaseQuantizer(nn.Module):
    # output = self.quant(input)
    # return self.dequantize(input)
