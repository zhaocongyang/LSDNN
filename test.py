# CUDA_VISIBLE_DEVICES=2 python test.py
import argparse
import os
import numpy as np

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser(description="PyTorch LSDNN TESTING")
parser.add_argument('--in-path', type=str, required=False, default = "./test_img/line6.jpg", help='image to test')
parser.add_argument('--out-path', type=str, required=False, default = "./test_img/result_line6.jpg", help='mask image to save')
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--ckpt', type=str, default= '/mnt/hdd1/zcy/Deeplab-run/pascal/deeplab-resnet/experiment_6/checkpoint.pth.tar',#'./run/pascal/deeplab-resnet/model_best.pth.tar',#'deeplab-resnet.pth.tar',
                    help='saved model')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--no-cuda', action='store_true', default=True, 
                    help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco', 'cityscapes'],
                    help='dataset name (default: pascal)')
parser.add_argument('--crop-size', type=int, default=513,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--test_folder', default='./data_light/JPEGImages/', type=str,
                    help='Dir to save results')
parser.add_argument('--save_testresult_folder', default='./test_img/result_light/', type=str,
                    help='Dir to save results')



def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    model = DeepLab(num_classes=21,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    # 打开TXT，获取待测试的图片名
    filename = args.test_folder + 'test.txt'
    file = open(filename)
    for line in file:
        # 生成输入图片的路径和存储的路径
        image_name = line.strip('\n')
        input_image = args.test_folder + image_name + '.jpg'
        print(input_image) 
        save_result = args.save_testresult_folder + 'result_' + image_name + '.jpg'
        print(save_result) 
        
        # 开始测试~~
        image = Image.open(input_image).convert('RGB')
        target = Image.open(input_image).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            image = image.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 
                                3, normalize=False, range=(0, 255))
        print("type(grid) is: ", type(grid_image))
        print("grid_image.shape is: ", grid_image.shape)
        save_image(grid_image, save_result)

if __name__ == "__main__":
   main()