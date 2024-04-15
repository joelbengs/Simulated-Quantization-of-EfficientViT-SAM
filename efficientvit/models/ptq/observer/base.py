# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch


class BaseObserver:

    def __init__(
        self,
        module_type,
        bit_type,
        calibration_mode,
        # arguments for statistical quantization analysis
        stage_id=None,
        block_name=None,
        block_position=None,
        layer_position=None,
        conv_is_attention_qkv=None,
        conv_is_attention_scaling=None,
        conv_is_attention_projection=None,
        block_is_bottleneck=None,
        block_is_neck=None,
        operation_type="unknown"
            ):
        
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = torch.finfo(torch.float32).eps

        self.stage_id = stage_id
        self.block_name = block_name
        self.block_position = block_position
        self.layer_position = layer_position
        self.block_is_bottleneck=block_is_bottleneck
        self.block_is_neck=block_is_neck
        self.conv_is_attention_qkv=conv_is_attention_qkv
        self.conv_is_attention_scaling=conv_is_attention_scaling
        self.conv_is_attention_projection=conv_is_attention_projection
        self.operation_type = operation_type

        self.stored_weight_tensor = None

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.module_type in ['conv_weight', 'linear_weight']:
            v = v.reshape(v.shape[0], -1)
        elif self.module_type == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
            v = v.reshape(-1, v.shape[-1])
            v = v.transpose(0, 1)
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError

    # TODO: Verify this code
    def store_tensor(self, tensor: torch.tensor):
        #if self.operation_type == 'conv_weight'
        self.stored_weight_tensor = tensor
        #else:
        #    raise NotImplementedError 