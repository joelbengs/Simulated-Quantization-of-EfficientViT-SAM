# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
# Modified by Joel Bengs on 2024-06-11 under Apache-2.0 license
# Changes made:
# - Implemented into EfficientViT-SAM for quantization simulation

'''
Class that exports a dictionary of BitType objects, including uint4, int8 and uint8
'''

class BitType:

    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        if not self.signed:
            return 2**self.bits - 1
        return 2**(self.bits - 1) - 1

    @property
    def lower_bound(self):
        if not self.signed:
            return 0
        return -(2**(self.bits - 1))

    @property
    def range(self):
        return 2**self.bits

    def update_name(self):
        self.name = ''
        if not self.signed:
            self.name += 'uint'
        else:
            self.name += 'int'
        self.name += '{}'.format(self.bits)

    def to_dict(self):
        return {
            'bits': self.bits,
            'signed': self.signed,
            'name': self.name,
            'upper_bound': self.upper_bound,
            'lower_bound': self.lower_bound,
            'range': self.range
        }

BIT_TYPE_LIST = [
    BitType(4, False, 'uint4'),
    BitType(8, True, 'int8'),
    BitType(8, False, 'uint8'),
    BitType(16, True, 'int16'), #custom. Not sure about validity
]

BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}
