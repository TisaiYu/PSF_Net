"""
@coding: utf-8
@File Name: train
@Author: TisaiYu
@Creation Date: 2024/1/10
@version: 2.0
@Modification Record:
-----------------------------------------------------------------------------------------------------------------------------------------
文件重要修改
1.手动删除Dataset返回的为1的batch_size，将cal_size**2展平放在第一维度作为batch_size
2.设置数据集的步长选项，可调节选取区域的步长，变量为args.stride
3.添加中间输出方差sigma
4. 3.0待，不应该输入aif，因为对于一个像素点，相同深度，相同对焦距离应该输出一样的sigma，但是由于aif不一样可能就不会了。
------------------------------------------------------------------------------------------------------------------------------------------
@Description:
-------------------------------------------------------------------------------------------------------------------------------------------
文件头注释
根据师兄的意思：
输入 全聚焦图，深度图，坐标和对焦距离（相机参数可要可不要，暂时没要） | 取图片的区域cal_size*cal_size进行计算 | 输出 有颜色的cal_size**2*k_size**2*3的kernel。
按每个像素点kernel上的每个值加到对应图片的坐标位置上，计算中心区域ac_size*ac_size的loss。 | 训练时时args.stride小于等于计算loss的区域大小ac_size，测试时args.stride等于cal_size

----------------------------------------------------------------------------------------------------------------------------------------

"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import time
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset.mydataset_batch_rays import  PSFAllsceneDataset
from model import PSF3dNet
from utils.args import  get_arg_parser
from utils.PSFutils import *
def train(args):
    # TisaiYu[2024/1/15] 处理数据集
    mydataset = PSFAllsceneDataset(args)
    train_dataset,valid_dataset = random_split(mydataset,[0.7,0.3])
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)

    # TisaiYu[2024/1/15] 先读取一个batch看下维度，来初始化模型
    dataiter = iter(train_loader)
    data,_,_1 = next(dataiter) # TisaiYu[2024/1/15] data的维度为(1,cal_size**2(计算的区域),7)，1为batch_size(后面手动删除并设置cal_size**2为batch_size），7为aif3通道+深度1通道+对焦距离1+坐标2u，v
    if args.color_kernel:
        model = PSF3dNet(data.shape[2], args.k_size**2*3,args) # TisaiYu[2024/1/15] 输入为7，输出为带颜色的核
    else:
        model= PSF3dNet(data.shape[2], args.k_size**2,args)
    model = model.cuda()
    print(model)

    # TisaiYu[2024/1/22] 一些训练设置
    criterion = nn.MSELoss()# TisaiYu[2024/1/11] 默认是均值的
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.001,lr=0.00001,)# TisaiYu[2024/1/11] SGD学习率为1e-5时loss会出现nan，设置小一点正常
    training = 1 # TisaiYu[2024/1/15] 只进行验证就为0
    start_epoch = 0 # TisaiYu[2024/1/15] 为了resume使epoch正确设置的变量

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=args.epochs,
    #                                                 steps_per_epoch=len(train_loader),
    #                                                 cycle_momentum=True,
    #                                                 base_momentum=0.85, max_momentum=0.95, last_epoch=-1,
    #                                                 div_factor=25,
    #                                                 final_div_factor=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # TisaiYu[2024/1/15] 加载以前的权重重新开始训练
    if args.resume ==1:
        model_dict = torch.load(r"E:\Postgraduate\CZ\24_01CZnerf_PSF_modified\no_aif_calsize127_ksize17_ac_size101\checkpoint-139.pth")
        state_dict = model_dict['model']
        optim_dict = model_dict['optimizer']
        model.load_state_dict(state_dict)
        start_epoch = model_dict['iter']
        optimizer.load_state_dict(optim_dict)


    # TisaiYu[2024/1/22] 训练

    for epoch in range(start_epoch,args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('-' * 10)
        if training:
            model.train()
            pbar = tqdm(train_loader)
            for i, (data,focus_seg,aif_) in enumerate(pbar): #第一个维度是batch为1，得squeeze处理
                optimizer.zero_grad()
                for param_group in optimizer.param_groups:
                    print(param_group['lr'])
                    # if epoch<500:
                    #     param_group['lr'] = 0.0001
                    # else:
                    #     param_group['lr'] = 0.00001
                data = data.squeeze(0).float().cuda()  # TisaiYu[2024/1/15] 因为Dataset类的设计原因，只能把第一维度为1的batch_size手动删除，把第二维度作为batch_size
                tmp_g_gauss, tmp_g_whether_color = model(data)

                # TisaiYu[2024/1/22] 模型输出带颜色的核 或者 只是权重，输出带颜色的核则要输入aif
                if args.input_aif:
                    aif_patch = data[:,4:].cuda()
                else:
                    aif_patch = aif_.float().cuda()
                if args.color_kernel:
                    out_focus_img = gauss_conv_corlor(tmp_g_whether_color, args.cal_size, args.k_size)
                else:
                    out_focus_img = gauss_conv_weight(tmp_g_whether_color, aif_patch,  args.cal_size,args.k_size)

                blur_g1 = gauss_conv_weight(tmp_g_gauss, aif_patch, args.cal_size,args.k_size)
                blur_g1 = torch.clip(blur_g1, 0, 255)
                out_focus_img = torch.clip(out_focus_img, 0, 255)
                # TisaiYu[2024/1/10] 计算中间小区域的loss
                idx_from = args.cal_size // 2 - args.ac_size // 2 # TisaiYu[2024/1/15] 计算loss的区域对应的左上角横坐标（方的，横纵相等的）
                idx_to = idx_from + args.ac_size # TisaiYu[2024/1/15] 计算loss的区域对应的右下角横坐标
                focus_seg = focus_seg.squeeze(0).float().cuda() # TisaiYu[2024/1/15] 原模糊图对应patch
                foc_seg_ac = focus_seg[idx_from:idx_to, idx_from:idx_to,:]
                psf_blur_ac = blur_g1[idx_from:idx_to, idx_from:idx_to,:]
                out_focus_img_ac = out_focus_img[idx_from:idx_to, idx_from:idx_to,:]

                # display_image(blur_g1,args.cal_size,args.cal_size)
                # display_image(out_focus_img,args.cal_size,args.cal_size)
                loss1 = criterion( psf_blur_ac,foc_seg_ac)
                loss2 = criterion( out_focus_img_ac,foc_seg_ac) # TisaiYu[2024/1/11] 开始loss很低0.6是因为图片区域恰好取到黑区域了，权重初始也接近0
                # if loss1.item()<50:
                #     display_image(psf_blur_ac, args.ac_size, args.ac_size)
                # print("add_time:", t3 - t1)
                if epoch < 500:
                    loss =loss2
                    pbar.set_description(
                        f"Train Loss {loss.item():.4f} = 0.9 * loss1:{loss1.item():.4f} + 0.1 * loss2:{loss2.item():.4f})")
                else:
                    loss = 0.1*loss1+0.9*loss2
                    pbar.set_description(
                        f"Train Loss {loss.item():.4f} = 0.1 * loss1:{loss1.item():.4f} + 0.9 * loss2:{loss2.item():.4f})")
                # 输出梯度
                # grads = {}
                # for name, param in model.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         grads[name] = param.grad
                # print(grads)



                loss.backward()
                optimizer.step()
                scheduler.step()




        if (epoch+1) % 10==0: # TisaiYu[2024/1/15] 验证并保存权重

            state = {
                'iter': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_dir=f"no_aif_calsize{args.cal_size}_ksize{args.k_size}_ac_size{args.ac_size}"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, f'checkpoint-{epoch}.pth')
            torch.save(state, save_path)
            model.eval()

            pbar_test = tqdm(valid_loader)
            for i, (data,focus_seg,aif_) in enumerate(pbar_test): #第一个维度是batch，得处理
                with torch.no_grad():
                    data = data.squeeze(0).float().cuda()  # TisaiYu[2024/1/15] 因为Dataset类的设计原因，只能把第一维度为1的batch_size手动删除，把第二维度作为batch_size
                    tmp_g_gauss, tmp_g_whether_color = model(data)

                    if args.input_aif:
                        aif_patch = data[:, 4:].cuda()
                    else:
                        aif_patch = aif_.float().cuda()
                    if args.color_kernel:
                        out_focus_img = gauss_conv_corlor(tmp_g_whether_color, args.cal_size, args.k_size)
                    else:
                        out_focus_img = gauss_conv_weight(tmp_g_whether_color, aif_patch, args.cal_size, args.k_size)

                    blur_g1 = gauss_conv_weight(tmp_g_gauss, aif_patch, args.cal_size,args.k_size)

                    # TisaiYu[2024/1/10] 计算中间小区域的loss
                    idx_from = args.cal_size // 2 - args.ac_size // 2 # TisaiYu[2024/1/15] 计算loss的区域对应的左上角横坐标（方的，横纵相等的）
                    idx_to = idx_from + args.ac_size # TisaiYu[2024/1/15] 计算loss的区域对应的右下角横坐标
                    focus_seg = focus_seg.squeeze(0).float().cuda() # TisaiYu[2024/1/15] 对应原模糊图patch
                    foc_seg_ac = focus_seg[idx_from:idx_to, idx_from:idx_to,:]
                    psf_blur_ac = blur_g1[idx_from:idx_to, idx_from:idx_to,:]
                    out_focus_img_ac = out_focus_img[idx_from:idx_to, idx_from:idx_to,:]
                    loss1 = criterion(psf_blur_ac,foc_seg_ac)
                    loss2 = criterion(out_focus_img_ac,foc_seg_ac) # TisaiYu[2024/1/11] 开始loss很低0.6是因为图片区域恰好取到黑区域了，权重初始也接近0

                    # print("add_time:", t3 - t1)
                    loss = 0.5*loss1+0.5*loss2
                    pbar_test.set_description(f"Val Loss {loss.item():.4f} = 0.5 * (loss1:{loss1.item():.4f} + loss2:{loss2.item():.4f})")
                        # print(test_out_focus_img)
                        # print("结果与原区域相减: ",abs(test_out_focus_img[idx_from:idx_to, idx_from:idx_to, :]-test_focus_seg[idx_from:idx_to, idx_from:idx_to, :]))
                        #
                        # exit()

        print("\n") # TisaiYu[2024/1/11] 为了输出格式好看




if __name__ == '__main__':

    args = get_arg_parser().parse_args()
    cudnn.benchmark = True

    train(args)