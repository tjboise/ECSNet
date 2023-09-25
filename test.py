# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""
import os
import time
import torch.nn as nn

from matplotlib import pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50, lraspp_mobilenet_v3_large
from tqdm import tqdm
from model.transformer_seg import SETRModel
from setr.SETR import SETR_PUP

from model.enet import ENet, ENet_improved, ENet_crop2
from utils.utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
from model.Models import AttU_Net, AttU_Net_min,R2AttU_Net, R2U_Net
import cv2
from model.unet_model import UNet
from model.FCN import FCN32s
from model.ESCNet_min import ESCNet_min
from model.DeepCrack import DeepCrack


# def cal_miou(test_dir="./images/cracks/Test_Images_DeepCrack",
#              pred_dir="./images/cracks/results", gt_dir="./images/cracks/Test_Labels_DeepCrack"):
def cal_miou(test_dir="./images/patholes/Test_Images",
             pred_dir="./images/patholes/results", gt_dir="./images/patholes/Test_Labels"):
# def cal_miou(test_dir="./images/cracks/Test_Images_CFD",
#              pred_dir="./images/cracks/results", gt_dir="./images/cracks/Test_Labels_CFD"):
# def cal_miou(test_dir="./images/cracks/Test_Images_GAPS384",
#              pred_dir="./images/cracks/results", gt_dir="./images/cracks/Test_Labels_GAPS384"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "pathole"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        # net = DeepCrack(num_classes=1)
        # net = ESCNet_min(num_classes=1, in_channels=1)
        # net = FCN32s(n_class=1)
        # net.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        # net = UNet(n_channels=1, n_classes=1)
        # net = R2AttU_Net(in_ch=1, out_ch=1, t=2)
        # net = R2U_Net(img_ch=1, output_ch=1, t=2)
        # net = ENet_improved(num_classes=1, in_channels=1)
        # net = ENet_crop2(num_classes=1, in_channels=1)
        # net = AttU_Net(img_ch=1, output_ch=1)

        # net = SETRModel(patch_size=(16, 16),
        #                 in_channels=1,
        #                 out_channels=1,
        #                 hidden_size=1024,
        #                 num_hidden_layers=6,
        #                 num_attention_heads=16,
        #                 decode_features=[512, 256, 128, 64])

        net = SETR_PUP(
            img_dim=256,
            patch_dim=16,
            num_channels=1,
            num_classes=1,
            embedding_dim=768,
            num_heads=12,
            num_layers=12,
            hidden_dim=3072,
            dropout_rate=0.1,
            attn_dropout_rate=0.1

        )

        # net = AttU_Net(img_ch=1, output_ch=1)
        # net = deeplabv3_resnet50(num_classes=1)
        # net = fcn_resnet50(num_classes=1)
        # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # net = ENet(num_classes=1, in_channels=1)
        #
        # net = lraspp_mobilenet_v3_large(num_classes=1)
        # # #
        # # # # net.classifier._modules['6'] = nn.Linear(4096, 4)#for vgg16, alexnet
        # # # net.backbone.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # for vgg16, alexnet
        # # #
        # net.backbone._modules['0']._modules['0'] = Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
        #                                                   bias=False)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model.pth', map_location=device)) # todo
        # 测试模式
        net.eval()
        print("Load model done.")

        # target_layers = [net.down4.maxpool_conv]


        img_names = os.listdir(test_dir)
        # print(img_names)
        image_ids=[]
        for image_name in img_names:
            if image_name[-3:]=='jpg':
                # print(image_name[:-4])
                image_ids.append(image_name[:-4])
                # image_ids =image_ids.append(image_name.split(".")[0])


        # image_ids = [image_name.split(".")[0] for image_name in img_names]

        # kk=0
        # writer = SummaryWriter('logs/test')


        print("Get predict result.")
        # with GradCAM(model=net, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        # times=0

        starttime = time.time()
        i=0

        for image_id in tqdm(image_ids):
            i=i+1

            image_path = os.path.join(test_dir, image_id + ".jpg")
            # print(image_path)
            label_path = os.path.join(gt_dir, image_id + ".png")
            # print(label_path)

            label = cv2.imread(label_path)
            img = cv2.imread(image_path)

            origin_shape = img.shape
            # print(origin_shape)
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label = cv2.resize(label, (256,256))
            label = (label/255).astype(int)
            # print(type(label))

            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # print(img_tensor.size())
            # 预测
            # starttime = time.time()
            pred = net(img_tensor)
            # endtime = time.time()
            # deltatime = endtime-starttime
            # times = times+deltatime

            # print(pred.size())
            # 提取结果
            #------------
            # pred=pred['out']
            #----------

            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            # print(origin_shape[1], origin_shape[0])
            # print(pred.size())
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

            # targets = [SemanticSegmentationTarget(0, label)]
            # grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0,:]
            # cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # plt.imshow(cam_image)
            # plt.show()
        endtime = time.time()




        print("Get predict result done.")
        print('==================')
        print('i=',i)
        print('total:', endtime-starttime)
        print('mean:',(endtime-starttime)/i)

        # print('total-time',times)
        # print('mean-time', times/i)

        print('==================')

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # 执行计算mIoU的函数

        print("Get miou done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()