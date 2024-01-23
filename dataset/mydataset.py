import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class PSFAllsceneDataset(Dataset):
    def __init__(self, data_folder, picture_nums_per_scene, scale, cal_size, ac_size,k_size):  # n为缩放比例, cal_size参与计算的区域大小，如50*50，ac_size实际PSF有效大小，小于cal_size如5*5
        self.data_folder = data_folder
        self.picture_nums_per_scene = picture_nums_per_scene
        focus_path = os.path.join(data_folder, 'focus')
        self.scene_list = os.listdir(focus_path)
        self.scale = scale
        self.cal_size = cal_size
        self.ac_size = ac_size
        self.k_size = k_size

        img_path = os.path.join(data_folder, 'all_in_focus/1.tif')
        img = cv2.imread(img_path, 0)
        self.h = img.shape[0] // self.scale
        self.w = img.shape[1] // self.scale
        self.patchs_h = self.h - self.cal_size + 1
        self.patchs_w = self.w - self.cal_size + 1
        self.patchs_num_per_pic = self.patchs_w * self.patchs_h



    def __len__(self): # TisaiYu[2024/1/10] 应该写每个点为最小单位，而不是一片区域为最小单位，因为这样点的集合不够batch_size的torch的Dataset可以控制其舍弃或者截断都可以。
        return self.picture_nums_per_scene * len(self.scene_list) * self.patchs_num_per_pic

    def __getitem__(self, idx):
        # print("idx:", idx)
        foc_range = np.arange(0, 200, 200 // (self.picture_nums_per_scene))

        # 图片下标
        foc_num = idx // ( self.patchs_h* self.patchs_w)  # 第几张图片，从零开始

        # 场景下标
        scene_idx = foc_num // self.picture_nums_per_scene  # 第几个场景，从零开始
        # print("第几个场景（从零开始）:",scene_idx)
        foc_idx = foc_num % self.picture_nums_per_scene  # 这张图片在场景内是第几张,从零开始
        # print("第几张图片（从零开始）:",foc_idx)
        foc_file_num = foc_range[foc_idx] + 1  # +1因为文件名是1开始的
        # print("图片对应文件名数字:", foc_file_num)
        patchs_in_pic = idx % self.patchs_num_per_pic #此张图片的第几个patch
        # print("图片的第几个patch（从零开始）:", patchs_in_pic)
        AIF_file = os.path.join(self.data_folder, r"all_in_focus\{}.tif".format(self.scene_list[scene_idx]))
        focus_file = os.path.join(self.data_folder, r"focus\{}\foc_{}.bmp".format(self.scene_list[scene_idx], foc_file_num))
        depth_file = os.path.join(self.data_folder, r"depth\{}\depth_new.npy".format(self.scene_list[scene_idx]))
        fd_file = os.path.join(self.data_folder, r"fd\{}\fd.npy".format(self.scene_list[scene_idx]))

        # print(AIF_file)
        # print(focus_file)
        # print(depth_file)
        # print(fd_file)

        fd = np.load(fd_file)
        focus_distance = fd[foc_file_num]
        # print(focus_distance)
        AIF = cv2.imread(AIF_file)
        AIF_r = cv2.resize(AIF, None, fx=1 / self.scale, fy=1 / self.scale)
        AIF_r = np.transpose(AIF_r, (2, 0, 1))
        focus = cv2.imread(focus_file)
        focus_r = cv2.resize(focus, None, fx=1 / self.scale, fy=1 / self.scale)
        focus_r = np.transpose(focus_r, (2, 0, 1))
        depth = np.load(depth_file)
        depth_r = cv2.resize(depth, None, fx=1 / self.scale, fy=1 / self.scale)

        patchs_row = patchs_in_pic // self.patchs_w
        patchs_col = patchs_in_pic % self.patchs_w

        aif_patch = AIF_r[:, patchs_row:patchs_row + self.cal_size, patchs_col:patchs_col + self.cal_size]
        focus_patch = focus_r[:, patchs_row:patchs_row + self.cal_size, patchs_col:patchs_col + self.cal_size]
        depth_patch = depth_r[patchs_row:patchs_row + self.cal_size, patchs_col:patchs_col + self.cal_size]
        xx,yy = torch.meshgrid(torch.linspace(0,self.w-1,self.w),torch.linspace(0,self.h-1,self.h),indexing="ij")
        xx = xx.t()
        yy = yy.t()
        xx_patch = xx[patchs_row:patchs_row + self.cal_size, patchs_col:patchs_col + self.cal_size]
        yy_patch = yy[patchs_row:patchs_row + self.cal_size,patchs_col:patchs_col + self.cal_size]
        data = np.zeros([7, depth_patch.shape[0], depth_patch.shape[1]])  # TisaiYu[2024/1/10] 每个像素点有像素颜色值3、深度1、对焦距离1、坐标u，v2（相机参数，每个像素宽高pmm，焦距fmm，f数可要可不要）

        data[:3,:,:] = aif_patch
        data[3,:,:] = depth_patch
        data[4,:,:] = focus_distance
        data[5,:,:] = xx_patch#通过(data[5,x,y],data[6,x,y])来访问(x,y)如设置x=21，y=22则(data[5,x,y],data[6,x,y])打印为(21,22)
        data[6,:,:] = yy_patch
        data = torch.Tensor(data).float().cuda()
        focus_patch = torch.Tensor(focus_patch).float().cuda()

        return data,focus_patch
