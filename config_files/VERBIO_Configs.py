class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1

        self.num_classes = 16
        self.embed_dim = 64

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.lr = 5e-4
        self.eta_min = 1e-5

        # data parameters
        self.drop_last = True
        self.batch_size = 64
        
        self.kernel_size = 5
        self.stride = 1
        self.embed_dim = 64
        self.tcn_nfilters = [48, 96, 192, 384, 64]
        self.tcn_kernel_size = 2
        self.tcn_dropout = 0.2
        self.trans_d_model = 512
        
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8

