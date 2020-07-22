# -*- coding: utf-8 -*
# CUDA_VISIBLE_DEVICES=1 python test_center_gpu_sensors.py
import argparse
import os
import numpy as np
import cv2
import time
import math
from line_profiler import LineProfiler

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--in-path', type=str, required=False, default = "./test_img/line6.jpg", help='image to test')
parser.add_argument('--out-path', type=str, required=False, default = "./test_img/result_line6.jpg", help='mask image to save')
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--ckpt', type=str, default='/mnt/hdd1/zcy/Deeplab-run/pascal/deeplab-resnet/experiment_4/checkpoint.pth.tar', 
                    help='saved model')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--no-cuda', action='store_true', default=False, 
                    help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
parser.add_argument('--dataset', type=str, default='pascal',
                    choices=['pascal', 'coco', 'cityscapes'],
                    help='dataset name (default: pascal)')
parser.add_argument('--crop-size', type=int, default=513,#513
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--test_folder', default='./data_sensors/JPEGImages/', type=str,
                    help='Dir to save results')
parser.add_argument('--save_testresult_folder', default='./test_img/result_sensor_paper/', type=str,
                    help='Dir to save results')

# 每一行都使用灰度重心法计算出光条中心，易于理解版本
# 速度2.1s
def Cvpointgray(img):
    gray_center = []
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print("img.shape[0] = ", img.shape[0])
    for i in range(img.shape[0]):
        gray_value = 0
        gray_coords = 0
        for j in range(1,img.shape[1]):
            gray_value = gray_value + img[i,j]
            gray_coords = gray_coords + img[i,j]*j
        # print("gray_value = ", gray_value)
        if gray_value < 2550:
            continue
        else:
            x = gray_coords/gray_value
            x = np.round(x)
            gray_center.append([x,i])
    return gray_center

# 每一行都使用灰度重心法计算出光条中心，
# 使用数组计算版本，速度远远快于之前的
# 速度 0.004s
def Cvpointgray_fast(img):
    # image size 838行，1000列 img.shape = (838,1000)
    height, width, channel = img.shape
    img_array = np.array(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    # 每一行灰度值求和，并将一些灰度值之和小的行置零，计算结果是灰度重心的分母，每一行都用灰度重心，因此分母的维度是 838 * 1
    sum_gravity = np.sum(img_array, axis=1)
    sum_gravity[np.where(sum_gravity < 2550)] = 0
    # 生成索引序列
    index = np.arange(1, width + 1) # 生成1-1000的数组 代表一行中的每一个像素的索引
    img_index = np.tile(index, height).reshape(height, width) # 重复838次，得到838行，1000列的数组，和图片大小一样
    # 索引和灰度值相乘，计算结果就是灰度重心法的分子
    img_idx_mul_gray = np.multiply(img_index,img_array)
    weight_sum_gray = np.sum(img_idx_mul_gray, axis=1)
    # 灰度*坐标／
    gray_center = np.divide(weight_sum_gray, sum_gravity)

    return gray_center

# 使用 GPU pytorch 计算版本，
# 处理的是sensor paper的图片，使用的是 列灰度重心
# 速度 
def Cvpointgray_gpu(image_tensor_gray, img_index):
    # image size 1080行，1920列 img.shape = (1080,1920)
    height, width = image_tensor_gray.size()
    # 每一列灰度值求和，并将一些灰度值之和小的行置零，计算结果是灰度重心的分母，每一行都用灰度重心，因此分母的维度是 838 * 1
    sum_gravity = torch.sum(image_tensor_gray, axis=0)
    # 索引和灰度值相乘，计算结果就是灰度重心法的分子
    img_idx_mul_gray = img_index.float() * image_tensor_gray.float()
    weight_sum_gray = torch.sum(img_idx_mul_gray, axis=0)
    # 灰度*坐标／灰度之和
    gray_center = torch.div(weight_sum_gray, sum_gravity.float())

    return gray_center #gray_center.cpu().numpy()

# from torch2trt import torch2trt

# # create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()

# # create example data
# x = torch.ones((1, 3, 224, 224)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])

def main():
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.cuda = torch.cuda.is_available()
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

    # 
    filename = args.test_folder + 'test.txt'
    file = open(filename)
    # init time parameters
    forward_min = math.inf
    forward_time = 0
    model_time_min = math.inf
    model_time = 0
    image_number = 0
    
    # init model cuda and image index
    model.eval()        
    image_width = 1920
    image_heigth = 1080
    if args.cuda:
        model.cuda() # 第一次挂载到CUDA需要消耗很长时间，大概4.1208秒
        index = torch.arange(1, image_heigth + 1).t().reshape(-1, 1) # 生成1-1080的数组 代表一行中的每一个像素的索引
        # print("index = ", index, index.size())
        img_index = index.repeat(1, image_width).cuda().float() # 重复1920次，得到1080行，1920列的数组
        # print("img_index = ", img_index[0], img_index, img_index.size())
    else:
        index = torch.arange(1, image_heigth + 1).t() # 生成1-1000的数组 代表一行中的每一个像素的索引
        img_index = index.repeat(image_width, 1).reshape(image_heigth, image_width) # 重复838次，得到838行，1000列的数组

    for line in file:
        # 生成输入图片的路径和存储的路径
        image_name = line.strip('\n')
        input_image = args.test_folder + image_name + '.jpg'
        print(input_image) 
        save_result = args.save_testresult_folder + 'result_' + image_name + '.jpg'
        print(save_result) 
        image_number = image_number + 1 
        
        # 开始测试~~
        image = Image.open(input_image).convert('RGB')
        target = Image.open(input_image).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)
        # 图片转换成数组，两个数组是因为需要大光斑（红色）和小光斑（绿色）
        image_array = np.array(image) # 存储大光斑（红色）
        image_array2 = np.copy(image_array) # 存储小光斑（绿色）
        image_tensor = torch.from_numpy(image_array)
        # image_tensor2 = torch.from_numpy(image_array2)


        # model.eval()        
        if args.cuda:
            # model.cuda() # 第一次挂载到CUDA需要消耗很长时间，大概4.1208秒
            tensor_in = tensor_in.cuda()
            image_tensor = image_tensor.cuda()
            # image_tensor2 = image_tensor2.cuda()
            
        
        start = time.time()

        with torch.no_grad():
            output = model(tensor_in)#.cpu()
            # convert to TensorRT feeding sample data as input
            # model_trt = torch2trt(model, [tensor_in])
            # output = model_trt(tensor_in).cpu()

        # find background index
        # torch.max output max value as dim 0, and index of max value as dim 1; use this we can get the class of each pixel
        image_tensor[torch.max(output[:3], 1)[1][0] != 1] = 0
        # get red channel and use red channel to calculate gray gravity
        image_tensor_gray =image_tensor[:,:,0] # red channel
       
        # save_result_roi = args.save_testresult_folder + 'result_roitemp' + image_name + '.jpg'
        # cv2.imwrite(save_result_roi, image_tensor_gray.cpu().numpy())

        elapsed = time.time() - start
        model_time_min = min(model_time_min, elapsed)
        model_time += elapsed

        # claculate gray center pixel
        # roi_image = cv2.merge([image_tensor.cpu().numpy()])
        # gray_center = Cvpointgray(roi_image)
        # gray_center = Cvpointgray_fast(roi_image)
        gray_center = Cvpointgray_gpu(image_tensor_gray, img_index)


        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed
        
        print("elapsed = ", elapsed)
        # 再读一次图片，用作展示
        image_orign = cv2.imread(input_image)#Image.open(input_image).convert('RGB')
        gray_center = gray_center.cpu().numpy()
        print("len(gray_center) = ", len(gray_center))
        for i in range(len(gray_center)):
            if np.isnan(gray_center[i]) or np.isinf(gray_center[i]):
                continue
            else:
                cv2.circle(image_orign, ( int(i), int(gray_center[i]) ), 2, (0,255,0))

        # 画大光条，易于理解的灰度重心法
        # for i in range(len(gray_center)):
        #     cv2.circle(image_orign, ( int(gray_center[i][0]), int(gray_center[i][1]) ), 2, (0,0,255))
       
        save_result_fin = args.save_testresult_folder + 'result_fin_' + image_name + '.jpg'
        cv2.imwrite(save_result_fin, image_orign)

        # test time
        # lp = LineProfiler()
        # lp_wrapper = lp(Cvpointgray_gpu)
        # lp_wrapper(image_tensor_gray, img_index)
        # lp.print_stats()

        # save class heatmap 
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()), 
                                3, normalize=False, range=(0, 255))
        print("type(grid) is: ", type(grid_image))
        print("grid_image.shape is: ", grid_image.shape)
        save_image(grid_image, save_result)
    
    forward_average = forward_time / image_number 
    model_average = model_time / image_number
    print('Forward: {0:.3f}/{1:.3f}'.format(forward_min, forward_average) , 's')
    print('Model: {0:.3f}/{1:.3f}'.format(model_time_min, model_average) , 's')


if __name__ == "__main__":
   main()

# from line_profiler import LineProfiler
# import random

# def do_sth(numbers):
#     s = sum(numbers)
#     for i in range(len(numbers)):
#         rr = numbers[i]/10
#         r = numbers[i]/s

# numbers = [random.randint(1,100) for i in range(1000)]
# lp = LineProfiler()
# lp_wrapper = lp(do_sth)
# lp_wrapper(numbers)
# lp.print_stats()