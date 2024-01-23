import cv2
import torch
import numpy as np
import torch.nn as nn

def gauss_conv_weight(tmp_g,aif,cal_size,k_size): # TisaiYu[2024/1/16] 这是原来根据每个像素点的权重核来计算最终结果的，还没有乘以颜色像素值，输入为通过sigma求出的权重,训练用，返回和cal_size区域大小相同
    tmp_g = tmp_g.view(cal_size, cal_size, k_size ** 2)
    # tmp_g = torch.zeros_like(tmp_g).float().cuda()
    # tmp_g[:, :, k_size // 2] = 0.000000000001
    aif =aif.view(cal_size, cal_size, 3)

    # wsum = torch.sum(tmp_g, dim=2).unsqueeze(2)  # TisaiYu[2024/1/16] tmp_g维度为（cal_size**2,k_size,k_size)
    # tmp_g = tmp_g / wsum  # 归一化哦

    out = torch.empty([cal_size, cal_size, 3]).float().cuda() # TisaiYu[2024/1/22] 返回cal_size**2大小
    output = torch.zeros([cal_size + k_size - 1, cal_size + k_size - 1, 3]).float().cuda() # TisaiYu[2024/1/22] 中间计算结果，最后只取其中cal_size**2大小部分
    mid = k_size // 2
    for j in range(k_size * k_size):
        hh = j // k_size
        ww = j % k_size

        output[hh:cal_size + hh, ww:ww + cal_size, :] += tmp_g[:, :, j:j+1]*aif

    out[:, :, :] = output[mid:mid + cal_size, mid:mid + cal_size, :]

    return out

def gauss_conv_weight_test(tmp_g,aif,cal_size,k_size): # TisaiYu[2024/1/16] 这是原来根据每个像素点的权重核来计算最终结果的，还没有乘以颜色像素值，输入为通过sigma求出的权重，测试用，返回和区域+k_size-1相同
    tmp_g = tmp_g.view(cal_size, cal_size, k_size ** 2)

    aif = aif.view(cal_size, cal_size, 3)  # TisaiYu[2024/1/16] aif维度为（cal_size**2,3)

    # wsum = torch.sum(tmp_g, dim=2).unsqueeze(2) # TisaiYu[2024/1/16] tmp_g维度为（cal_size**2,k_size,k_size)
    # tmp_g = tmp_g / wsum  # 归一化哦


    output = torch.zeros([cal_size + k_size - 1, cal_size + k_size - 1, 3]).float().cuda()

    for j in range(k_size * k_size):
        hh = j // k_size
        ww = j % k_size

        output[hh:cal_size + hh, ww:ww + cal_size, :] += tmp_g[:, :, j:j + 1] * aif  # 切片j:j+1可以保持这一维度为1，不会消除


    return output

def gauss_conv_corlor(color_kernel,cal_size,k_size): # TisaiYu[2024/1/10] color_kernel是已经kernel里面每个值乘了AIF对应点的了，所以为三通道
    color_kernel = color_kernel.view(cal_size,cal_size,k_size**2,3)


    out = torch.empty([cal_size,cal_size,3]).float().cuda()
    output = torch.zeros([cal_size+k_size-1,cal_size+k_size-1,3]).float().cuda()
    mid = k_size//2
    for j in range(k_size * k_size):
        hh = j // k_size
        ww = j % k_size


        output[  hh:cal_size + hh, ww:ww + cal_size,:] += color_kernel[:,:,j,:]

    out[ :, :, :] = output[ mid:mid + cal_size, mid:mid + cal_size,:]


    return out

def gauss_conv_color_test(color_kernel,cal_size,k_size): # TisaiYu[2024/1/10] color_kernel是已经kernel里面每个值乘了AIF对应点的了，所以为三通道
    color_kernel = color_kernel.view(cal_size,cal_size,k_size**2,3)


    output = torch.zeros([cal_size+k_size-1,cal_size+k_size-1,3]).float().cuda()

    for j in range(k_size * k_size):
        hh = j // k_size
        ww = j % k_size

        output[hh:cal_size + hh, ww:ww + cal_size, :] += color_kernel[ :,:, j,:]  # 切片j:j+1可以保持这一维度为1，不会消除

    return output





def gaussian_layer(cal_size, sigma, mu=torch.Size([]), rho=torch.Size([]),kernel_size=15): # TisaiYu[2024/1/16] 用于计算权重的，改掉了原来的value计算(师兄说不对)，用上面的gaussain_conv计算了
    ray_batch_num = cal_size**2
    X = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
    Y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
    Y_grid,X_grid = torch.meshgrid(X, Y,indexing="ij")
    X_grids = torch.tile(X_grid[None, ...], [ray_batch_num, 1, 1]).cuda()
    Y_grids = torch.tile(Y_grid[None, ...], [ray_batch_num, 1, 1]).cuda()
    sigma1, sigma2 = sigma[:, 0], sigma[:, 1]
    # imgB = torch.zeros((rays_batch_num, 3))

    if mu != torch.Size([]) and rho != torch.Size([]):
        mu1, mu2 = mu[:, 0], mu[:,1]  # TisaiYu[2024/1/16] 下面的None表示拓展维度为1，相当于unsqueeze()，也就是None的地方全是1了，sigma1[..., None, None]原来是1维，现在就是3维了。
        a = 1. / (2. * torch.pi * sigma1[..., None, None] * sigma2[..., None, None] * (
            torch.sqrt(1 - rho[..., None, None] * rho[..., None, None])))
        b = -1. / (2. * (1 - rho[..., None, None] * rho[..., None, None])) * (
                (X_grids[..., :, :] - mu1[..., None, None]) * (
                X_grids[..., :, :] - mu1[..., None, None])
                / (sigma1[..., None, None] * sigma1[..., None, None])
                - 2 * rho[..., None, None] * (X_grids - mu1[..., None, None]) * (
                        Y_grids[..., :, :] - mu2[..., None, None])
                / (sigma1[..., None, None] * sigma2[..., None, None])
                + (Y_grids[..., :, :] - mu2[..., None, None]) * (
                        Y_grids[..., :, :] - mu2[..., None, None])
                / (sigma2[..., None, None] * sigma2[..., None, None]))
        g = (a * (torch.exp(b))).type(torch.FloatTensor).cuda()
        # print(g)
        assert not torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)), print("a:", a, "b:", b)
    else:
        a = 1. / (2. * torch.pi * sigma1[..., None, None] * sigma2[..., None, None])
        b = -1. / 2. * ((X_grids[..., :, :] * X_grids[..., :, :]) / (
                sigma1[..., None, None] * sigma1[..., None, None])
                        + (Y_grids[..., :, :] * Y_grids[..., :, :]) / (
                                sigma2[..., None, None] * sigma2[..., None, None]))
        g = (a * (torch.exp(b))).type(torch.FloatTensor).cuda()
        # print(g)
        assert not torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)), print("a:", a, "b:", b)
    return g

def display_image(data, w, h):
    # 检查数据类型并转换为numpy数组
    if isinstance(data, torch.Tensor):
        # 如果Tensor在GPU上，先移动到CPU
        if data.is_cuda:
            data = data.cpu()
        # 转换为numpy数组
        data = data.detach().numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError("输入数据既不是PyTorch Tensor也不是numpy数组")

    # 确保数据是float或uint8类型
    if data.dtype  in [np.float32, np.float64] and data.ndim!=2:
        data = data.astype(np.uint8)



    # 调整形状和维度
    if data.ndim == 2:  # 深度图
        # 确保形状是(w, h)
        data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        data = data.astype(np.uint8)
        data = np.reshape(data, (h, w))
        # 显示深度图
        cv2.imshow('Depth Image', data)
    elif data.ndim == 3:  # 彩色图
        # 确保形状是(h, w, channels)
        if data.shape[0] in [3, 4]:  # 如果channels在第一个维度
            data = np.transpose(data, (1, 2, 0))

        # 如果是RGBA，转换为RGB
        if data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
        # 显示彩色图
        print(data.shape)
        cv2.imshow('Color Image', data)
    else:
        raise ValueError("图像数据的维度不正确")

    # 等待键盘事件
    cv2.waitKey(0)
    cv2.destroyAllWindows()