# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented into EfficientViT-SAM for quantization simulationfrom .log2 import Log2Quantizer
from .uniform import UniformQuantizer
from .log2 import Log2Quantizer

str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}

def build_quantizer(quantizer_str, bit_type, observer, module_type):
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
