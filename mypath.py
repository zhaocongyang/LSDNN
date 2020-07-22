class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/zcy/Documents/Deeplab-xception-pytorch/data_sensors/' #'/mnt/hdd1/dataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/home/zcy/Documents/Deeplab-xception-pytorch/dataloaders/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/mnt/hdd1/dataset/MSCOCO2017/'
        elif dataset == 'my':
            return '/home/zcy/Documents/Deeplab-xception-pytorch/data/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
