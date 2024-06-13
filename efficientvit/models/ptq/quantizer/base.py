# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented into EfficientViT-SAM for quantization simulation
import torch.nn as nn

class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    # abstract
    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    # abstract
    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    # Here it is, the most low level forward method! Model inference cases a chain of forward methods, all the way down here.
    # the incoming weight is quantized, then dequantized again - all to cause artificall information loss
    def forward(self, inputs):
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)
        return outputs
