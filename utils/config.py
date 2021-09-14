from pprint import pprint
from yacs.config import CfgNode as CN

# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'
cfg = CN()


cfg.min_size = 600  # image resize
cfg.max_size = 1000 # image resize
cfg.num_workers = 8
cfg.test_num_workers = 8

# sigma for l1_smooth_loss
cfg.rpn_sigma = 3.
cfg.roi_sigma = 1.

# param for optimizer
# 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
cfg.weight_decay = 0.0005
cfg.lr_decay = 0.1  # 1e-3 -> 1e-4
cfg.lr = 1e-3


# visualization
cfg.env = 'faster-rcnn'  # visdom env
cfg.port = 8097
cfg.plot_every = 40  # vis every N iter

# preset
cfg.data = 'voc'
cfg.pretrained_model = 'vgg16'

# training
cfg.epoch = 14

cfg.use_adam = False # Use Adam optimizer
cfg.use_chainer = False # try match everything as chainer
cfg.use_drop = False # use dropout in RoIHead
# debug
cfg.debug_file = './debugf'

cfg.test_num = 10000
# model
cfg.load_path = None

cfg.caffe_pretrain = False # use caffe pretrained model instead of torchvision
cfg.caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

## datasets
cfg.VOC_BBOX_LABEL_NAMES=(
        #"__background__ ",# 0
        'passenger ship',  # 1
        'ore carrier',  # 2
        'general cargo ship',  # 3
        'fishing boat',  # 4
        'Sail boat',  # 5
        'Kayak',  # 6
        'flying bird',  # flying bird/plane #7
        'vessel',  # vessel/ship #8
        'Buoy',  # 9
        'Ferry',  # 10
        'container ship',  # 11
        'Other',  # 12
        'Boat',  # 13
        'Speed boat',  # 14
        'bulk cargo carrier',  # 15
    )


cfg.voc_data_dir='E:/fjj/SeaShips_SMD'
cfg.split='label3'
cfg.DATASETS=CN()

# DATASETS = {
#         "coco_2017_train": {
#             "img_dir": "coco/train2017",
#             "ann_file": "coco/annotations/instances_train2017.json"
#         },}
# VOC_BBOX_LABEL_NAMES = (
#     'aeroplane',
#     'bicycle',
#     'bird',
#     'boat',
#     'bottle',
#     'bus',
#     'car',
#     'cat',
#     'chair',
#     'cow',
#     'diningtable',
#     'dog',
#     'horse',
#     'motorbike',
#     'person',
#     'pottedplant',
#     'sheep',
#     'sofa',
#     'train',
#     'tvmonitor')

class Config:
    # data
    voc_data_dir = 'E:/fjj/SeaShips_SMD'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = './debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
