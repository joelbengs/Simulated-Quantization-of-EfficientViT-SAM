from efficientvit.models.ptq import BIT_TYPE_DICT

class Config:

    def __init__(self, quant_method_W='minmax', quant_method_A='minmax'):
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''
        self.BIT_TYPE = BIT_TYPE_DICT['int8'] # default
        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.TEST_STR = "Config did get through to first QConvLayer!"
        self.TEST_STR_RESBLOCK = "Config did get through to ResBlock via Kwargs!"

        self.OBSERVER_STR = quant_method_W # default
        self.OBSERVER_W = quant_method_W
        self.OBSERVER_A = quant_method_A

        self.QUANTIZER_STR='uniform' # default
        self.QUANTIZER_W = 'uniform'
        self.QUANTIZER_A = 'uniform'
        self.QUANTIZER_A_LN = 'uniform'

        self.CALIBRATION_MODE='layer_wise'
        self.CALIBRATION_MODE_W = 'channel_wise'
        self.CALIBRATION_MODE_A = 'layer_wise'
        self.CALIBRATION_MODE_S = 'layer_wise'

        ptf=False
        lis=False
        if lis:
            self.INT_SOFTMAX = True
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
            self.OBSERVER_S = 'minmax'
            self.QUANTIZER_S = 'log2'
        else:
            self.INT_SOFTMAX = False
            self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
            self.OBSERVER_S = self.OBSERVER_A
            self.QUANTIZER_S = self.QUANTIZER_A
        if ptf:
            self.INT_NORM = True
            self.OBSERVER_A_LN = 'ptf'
            self.CALIBRATION_MODE_A_LN = 'channel_wise'
        else:
            self.INT_NORM = False
            self.OBSERVER_A_LN = self.OBSERVER_A
            self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
