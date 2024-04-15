from efficientvit.models.ptq import BIT_TYPE_DICT



class Config:

    def __init__(self, args):

        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']
        self.BIT_TYPE_N = BIT_TYPE_DICT['int8']

        # choices=["minmax", "ema", "omse", "percentile"]
        self.OBSERVER_W = args.observer_method_W if args.observer_method_W is not None else 'minmax'
        self.OBSERVER_A = args.observer_method_A if args.observer_method_A is not None else 'minmax'
        self.OBSERVER_N = args.observer_method_N if args.observer_method_N is not None else 'minmax'

        # choices=["uniform", "log2"]
        #self.QUANTIZER_W = args.quantize_method_W if hasattr(args, 'quantize_method_W') else 'uniform'
        self.QUANTIZER_W = args.quantize_method_W if args.quantize_method_W is not None else 'uniform'
        self.QUANTIZER_A = args.quantize_method_A if args.quantize_method_A is not None else 'uniform'
        self.QUANTIZER_N = args.quantize_method_N if args.quantize_method_N is not None else 'uniform'

        # choices=['layer_wise', 'channel_wise']
        self.CALIBRATION_MODE_W = args.calibration_mode_W if args.calibration_mode_W is not None else 'layer_wise'
        self.CALIBRATION_MODE_A = args.calibration_mode_A if args.calibration_mode_A is not None else 'layer_wise'
        self.CALIBRATION_MODE_N = args.calibration_mode_N if args.calibration_mode_N is not None else 'layer_wise'

    def to_dict(self):
        return vars(self)
    
        '''
        ptf stands for Power-of-Two Factor activation quantization for Integer Layernorm.
        lis stands for Log-Int-Softmax.
        These two are proposed in our "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer".
        '''

        # self.OBSERVER_STR = observer_method_W # default
        # self.OBSERVER_W = observer_method_W
        # self.OBSERVER_A = observer_method_W #TODO fix

        # self.QUANTIZER_STR='uniform' # default
        # self.QUANTIZER_W = 'uniform'
        # self.QUANTIZER_A = 'uniform'
        # self.QUANTIZER_A_LN = 'uniform'

        # self.CALIBRATION_MODE= 'layer_wise'
        # self.CALIBRATION_MODE_W = 'layer_wise' # 'channel_wise'
        # self.CALIBRATION_MODE_A = 'layer_wise'
        # self.CALIBRATION_MODE_S = 'layer_wise

        # # Remains from FQ-VIT
        # ptf=False
        # lis=False
        # if lis:
        #     self.INT_SOFTMAX = True
        #     self.BIT_TYPE_S = BIT_TYPE_DICT['uint4']
        #     self.OBSERVER_S = 'minmax'
        #     self.QUANTIZER_S = 'log2'
        # else:
        #     self.INT_SOFTMAX = False
        #     self.BIT_TYPE_S = BIT_TYPE_DICT['uint8']
        #     self.OBSERVER_S = self.OBSERVER_A
        #     self.QUANTIZER_S = self.QUANTIZER_A
        # if ptf:
        #     self.INT_NORM = True
        #     self.OBSERVER_A_LN = 'ptf'
        #     self.CALIBRATION_MODE_A_LN = 'channel_wise'
        # else:
        #     self.INT_NORM = False
        #     self.OBSERVER_A_LN = self.OBSERVER_A
        #     self.CALIBRATION_MODE_A_LN = self.CALIBRATION_MODE_A
