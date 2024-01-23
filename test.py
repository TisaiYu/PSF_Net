import math

import torch
import os
import numpy as np
import torch.nn as nn
import cv2
from model import PSF3dNet
from utils.args import  get_arg_parser
from utils.PSFutils import *



def test(pth_path,args,data_folder,p):

    # TisaiYu[2024/1/15] 定义模型和加载权重
    model = PSF3dNet(4 , args.k_size ** 2 ,args)
    model_dict = torch.load(pth_path)
    state_dict = model_dict['model']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()



    # TisaiYu[2024/1/15] 下面读取测试所需的全聚焦图，深度图，模糊图，对焦距离等文件
    focus_path = os.path.join(data_folder, 'focus')
    scene_list = os.listdir(focus_path) # TisaiYu[2024/1/15] 多少个场景
    focus_path = [os.path.join(data_folder, r"focus\{}".format(scene)) for scene in scene_list]
    args.picture_nums_per_scene = 20  # TisaiYu[2024/1/15] 每个场景取的图片数量，训练时为5，这里测试，则测试多点
    foc_range = np.arange(0, 200, 200 // (args.picture_nums_per_scene)) # TisaiYu[2024/1/15] 0到200，间隔args.picture_nums_per_scene取一张图片


    AIF_files = [os.path.join(data_folder, r"all_in_focus\{}.tif".format(scene)) for scene in scene_list ]
    focus_files = []
    for i in range(len(scene_list)): # TisaiYu[2024/1/15] 除了模糊图一个场景多张图片，其他都是一个场景一个文件。
        focus_files.append([os.path.join(focus_path[i], r"foc_{}.bmp".format(foc_num+1)) for foc_num in foc_range])
    depth_files = [os.path.join(data_folder, r"depth\{}\depth_new.npy".format(scene))for scene in scene_list]
    fd_files = [os.path.join(data_folder, r"fd\{}\fd.npy".format(scene))for scene in scene_list]


    #对每张图片进行处理，具体为缩放后，分割每张图为各个区域，取区域进行计算，最后把各个区域拼接为一张图
    for i in range(1):
        aif = cv2.imread(AIF_files[i])
        aif = cv2.resize(aif, None, fx=1 / args.scale, fy=1 / args.scale)
        depth = np.load(depth_files[i])
        depth = cv2.resize(depth, None, fx=1 / args.scale, fy=1 / args.scale)/1000.0
        fds = np.load(fd_files[i])
        img_h = depth.shape[0]
        img_w = depth.shape[1]
        padding_cha = [(0,args.cal_size - img_h%args.cal_size),(0,args.cal_size - img_w%args.cal_size),(0,0)]
        padding = [(0,args.cal_size - img_h%args.cal_size),(0,args.cal_size - img_w%args.cal_size)]
        aif = np.pad(aif, padding_cha, "constant") # TisaiYu[2024/1/19] 保持测试输出维度为原图大小才填充，深度图为了获得对应范围也填充了，不然不对应，默认填充0
        depth = np.pad(depth, padding, "constant") # TisaiYu[2024/1/19] 默认填充0
        h = depth.shape[0]
        w = depth.shape[1]


        patchs_h =(h - args.cal_size) // args.test_stride + 1  # TisaiYu[2024/1/15] 用作除数的都从1开始，被除数从零开始,包含了最后不足cal_size的部分，所以原图要padding
        patchs_w = (w - args.cal_size) // args.test_stride + 1
        patchs_num_per_pic = patchs_w * patchs_h  # TisaiYu[2024/1/15] 计算是从零开始的，计数就要加1

        # kernel = torch.empty([20, patchs_h*args.cal_size, patchs_w*args.cal_size, args.k_size ** 2])
        for j in range(20):

            focus = cv2.imread(focus_files[i][j])
            focus = cv2.resize(focus, None, fx=1 / args.scale, fy=1 / args.scale)
            focus = np.pad(focus, padding_cha, "constant")  # TisaiYu[2024/1/19] 默认填充0
            focus_distance = fds[foc_range[j]]/1000.0


            result_inplace = np.zeros([patchs_h*args.cal_size+args.k_size-1,patchs_w*args.cal_size+args.k_size-1,3])
            result_inplace_mid= np.zeros_like(result_inplace)
            result_inplace = torch.Tensor(result_inplace).float().cuda()
            result_inplace_mid = torch.Tensor(result_inplace_mid).float().cuda()


            for patchs_in_pic in range(patchs_num_per_pic):
                print(patchs_in_pic)

                with torch.no_grad():
                    patchs_row = patchs_in_pic // patchs_w
                    patchs_col = patchs_in_pic % patchs_w
                    patchs_row = patchs_row * args.test_stride # TisaiYu[2024/1/15] 训练步长是小于等于ac_size，测试步长应该是cal_size。所以这里是乘args.cal_size
                    patchs_col = patchs_col * args.test_stride
                    aif_patch = aif[patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size,:]
                    focus_patch = focus[patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size,:]
                    depth_patch = depth[patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size]
                    xx, yy = torch.meshgrid(torch.linspace(-(w // 2), w // 2, w),torch.linspace(-(h // 2), h // 2 - 1, h), indexing="ij")
                    xx = xx.t() # TisaiYu[2024/1/15] 转置后才是对应的坐标。
                    yy = yy.t()
                    xx_patch = xx[patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size]
                    yy_patch = yy[patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size]
                    data = np.zeros([args.cal_size * args.cal_size, 7])
                    data[:, 0] = depth_patch.reshape(args.cal_size * args.cal_size)
                    data[:, 1] = focus_distance
                    data[:, 2] = xx_patch.reshape(args.cal_size * args.cal_size)  # 通过(data[x][0],data[y_flatten,1])来访问(x,y)如设置x=21，y=22,y_flatten=y*cal_size得到(21,22)
                    data[:, 3] = yy_patch.reshape(args.cal_size * args.cal_size)
                    if args.input_aif:
                        data[:, 4:] = aif_patch.reshape(args.cal_size * args.cal_size, -1)
                    else:
                        data = data[:, 0:4]
                    data = torch.Tensor(data).float().cuda()
                    aif_patch = torch.Tensor(aif_patch).cuda()

                    if args.color_kernel:
                        tmp_g, color_kernel = model(data)
                        test_out_focus_img = gauss_conv_color_test(color_kernel, args.cal_size, args.k_size)
                        result_inplace[patchs_row:patchs_row + args.cal_size + args.k_size - 1,patchs_col:patchs_col + args.cal_size + args.k_size - 1, :] += test_out_focus_img

                    else:
                        tmp_g,tmp_g_color = model(data)
                        test_out_focus_img = gauss_conv_weight_test(tmp_g_color,aif_patch,args.cal_size, args.k_size)
                    blur_g1 = gauss_conv_weight_test(tmp_g, aif_patch,  args.cal_size,args.k_size)
                    blur_g1 = torch.clip(blur_g1, 0, 255)
                    test_out_focus_img = torch.clip(test_out_focus_img, 0, 255)

                    idx_from_pad = (args.cal_size+args.k_size-1)//2 - args.ac_size//2
                    idx_to_pad = idx_from_pad + args.ac_size

                    # display_image(result_inplace,args.cal_size+args.k_size-1,args.cal_size+args.k_size-1)
                    criterion = nn.MSELoss()
                    loss_ac = criterion(blur_g1[idx_from_pad:idx_to_pad,idx_from_pad:idx_to_pad,:],torch.Tensor(focus_patch[idx_from_pad:idx_to_pad,idx_from_pad:idx_to_pad,:]).float().cuda())
                    print("loss_ac:",loss_ac.item())
                    # display_image(blur_g1[idx_from_pad:idx_to_pad,idx_from_pad:idx_to_pad,:],args.ac_size,args.ac_size)

                    result_inplace[patchs_row:patchs_row + args.cal_size + args.k_size - 1,
                    patchs_col:patchs_col + args.cal_size + args.k_size - 1, :] += test_out_focus_img
                    result_inplace_mid[patchs_row:patchs_row + args.cal_size + args.k_size - 1,
                    patchs_col:patchs_col + args.cal_size + args.k_size - 1, :] += blur_g1

                    # kernel[j, patchs_row:patchs_row + args.cal_size, patchs_col:patchs_col + args.cal_size,
                    # :] = tmp_g.view(args.cal_size, args.cal_size, args.k_size ** 2)
                    # display_image(result_inplace_mid,patchs_h*args.cal_size,patchs_w*args.cal_size)
                    # TisaiYu[2024/1/15] 合为一张图，就是在一张图的每个patch区域通过模型后，依次拼在一个大的(h,w,k_size**2)上，这里的result_inplace就是(h+k_size-4,w+k_size-1,3),但是不是完全的h,w只是被整除的部分，最后batch不足的在训练时目前是舍弃的


                    # cv2.imshow("re",result_inplace_mid.cpu().numpy().astype(np.uint8))
                    # cv2.imshow("re1",result_inplace_mid.cpu().numpy().astype(np.uint8))


            mid = args.k_size//2
            result = result_inplace[mid :mid+img_h,mid:mid+img_w,:]
            result_mid = result_inplace_mid[mid :mid+img_h,mid:mid+img_w,:]
            print(result.shape)
            criterion = nn.MSELoss()
            focus_ori = torch.Tensor(focus[mid :mid+img_h,mid:mid+img_w,:]).float().cuda()
            loss = criterion(result_mid,focus_ori).item()
            print("loss:", loss)


            result = result_inplace[mid:mid + img_h, mid:mid + img_w, :].cpu().numpy().astype(np.uint8)  # TisaiYu[2024/1/16] 这个是刚好原图的形状区域
            result_mid = result_inplace_mid[mid:mid + img_h, mid:mid + img_w, :].cpu().numpy().astype(np.uint8)  # TisaiYu[2024/1/16] 这个是刚好原图的形状区域
            # cv2.imshow(f"result_calsize{args.cal_size}_ksize{args.k_size}_ac_size{args.ac_size}-focsimu{focus_files[i][j]}",result)
            # cv2.imshow(f"result_calsize{args.cal_size}_ksize{args.k_size}_ac_size{args.ac_size}-focmid{focus_files[i][j]}",result_mid)
            # cv2.waitKey()
            save_dir = f"test_result/pth{p}/scene{scene_list[i]}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file1 = os.path.join(save_dir,f"foc_{foc_range[j]+1}.png")
            save_file2 = os.path.join(save_dir,f"mid_foc_{foc_range[j]+1}.png")
            cv2.imwrite(save_file1,result)
            cv2.imwrite(save_file2, result_mid)

        for i in range(5):
            print(torch.sum(abs(kernel[0,:,:,:]-kernel[1,:,:,:])))






if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    test(r"E:\Postgraduate\CZ\24_01CZnerf_PSF_modified\no_aif_calsize127_ksize17_ac_size101\checkpoint-739.pth",args,r"E:\Postgraduate\dataset",14)

