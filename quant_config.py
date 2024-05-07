from efficientvit.models.ptq import BIT_TYPE_DICT



class Config:

    def __init__(self, args):

        self.BIT_TYPE_W = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_N = BIT_TYPE_DICT['int8']
        self.BIT_TYPE_A = BIT_TYPE_DICT['uint8']

        # choices=["minmax", "ema", "omse", "percentile"]
        self.OBSERVER_W = args.observer_method_W if args.observer_method_W is not None else 'minmax'
        self.OBSERVER_N = args.observer_method_N if args.observer_method_N is not None else 'minmax'
        self.OBSERVER_A = args.observer_method_A if args.observer_method_A is not None else 'minmax'

        # choices=["uniform", "log2"]
        #self.QUANTIZER_W = args.quantize_method_W if hasattr(args, 'quantize_method_W') else 'uniform'
        self.QUANTIZER_W = args.quantize_method_W if args.quantize_method_W is not None else 'uniform'
        self.QUANTIZER_N = args.quantize_method_N if args.quantize_method_N is not None else 'uniform'
        self.QUANTIZER_A = args.quantize_method_A if args.quantize_method_A is not None else 'uniform'

        # choices=['layer_wise', 'channel_wise']
        self.CALIBRATION_MODE_W = args.calibration_mode_W if args.calibration_mode_W is not None else 'layer_wise'
        self.CALIBRATION_MODE_N = args.calibration_mode_N if args.calibration_mode_N is not None else 'layer_wise'
        self.CALIBRATION_MODE_A = args.calibration_mode_A if args.calibration_mode_A is not None else 'layer_wise'


    def to_dict(self):
        return vars(self)