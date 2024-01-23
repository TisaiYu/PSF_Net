import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Training Config")
    #--data_settings
    parser.add_argument('--pic_per_scene', type=int, default=20)
    parser.add_argument('--data_path', type=str, default=r'E:\Postgraduate\dataset')
    parser.add_argument('--scale', type=int, default=4)


    #--training setttings
    parser.add_argument('--k_size', type=int, default=17)# TisaiYu[2024/1/10] 每个像素点的PSF核大小，这个应该设置大一点，尽可能大
    parser.add_argument('--cal_size', type=int, default=127)# TisaiYu[2024/1/10] train时的batch大小，有cal_size**2个点训练，一定不能设置shuffle=True，影响比较大
    parser.add_argument('--ac_size', type=int, default=101)# TisaiYu[2024/1/11] 计算loss时的中心区域大小，小于cal_size，只有训练时有用，测试时没这个
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1)# TisaiYu[2024/1/11] 由于getitem返回一个点每次加载太慢，所以返回一片区域，用cal_size来代替传统batch_size的作用，batch_size只能设置为1了就,然后手动将cal_size**2作为第一维度batch_size
    parser.add_argument('--stride', type=int, default=207)  # TisaiYu[2024/1/11] 获取数据集时，每次选取区域移到的步长，默认设置和ac_size一样大
    parser.add_argument('--resume', type=int, default=1)
    parser.add_argument('--embed', type=int, default=0)
    parser.add_argument('--color_kernel', type=int, default=0) # TisaiYu[2024/1/22] 是否输出带颜色核，和input_aif设置为一样
    parser.add_argument('--input_aif', type=int, default=0) # TisaiYu[2024/1/22] 是否输入aif进网络



    # --save setttings
    parser.add_argument('--model_savepath', type=str, default="./result")

    # --testing setttings
    parser.add_argument('--test_stride', type=int,default=127)  # TisaiYu[2024/1/15] 测试的步长，应该等于cal_size

    return parser
