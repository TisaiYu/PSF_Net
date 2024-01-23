import torch.nn as nn
import torch
from utils.PSFutils import gaussian_layer,gauss_conv_weight

class PositionEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):

        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)

# print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")


class PSF3dNet(nn.Module):
    def __init__(self,input_size,output_size,args):
        super(PSF3dNet, self).__init__()
        self.num_wide = 512
        self.num_hidden = 3
        self.args = args
        self.depth_embed_cnl = input_size if self.args.embed == 0 else get_embedder(10,)





        # self.pe = get_embedder(128,-1,input_size)
        self.flatten = nn.Flatten()
        # self.mlp_b =self.mlp_a(,3)
        # self.mlp_c =self.mlp_a(3)

        hiddens = [nn.Linear( self.num_wide, self.num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((self.num_hidden) * 2)]
        ### 参数化
        # first MLP
        self.linears1_1 = nn.Sequential(
            nn.Linear(self.depth_embed_cnl, self.num_wide), nn.ReLU(),
            *hiddens,
        )
        # self.linears1_2 = nn.Sequential(
        #     nn.Linear( self.num_wide, self.num_wide), nn.ReLU(),
        #     *hiddens,
        # )
        # self.linears1_3 = nn.Sequential(
        #     nn.Linear( self.num_wide, self.num_wide), nn.ReLU(),
        #     nn.Linear( self.num_wide, self.num_wide), nn.ReLU(),
        # )
        self.sigma_activate = nn.Sequential(
            nn.Linear(self.num_wide, self.num_wide // 2), nn.ReLU(),
            nn.Linear(self.num_wide // 2, 2)
        )
        # second MLP
        # self.linears2_1 = nn.Sequential(
        #     nn.Linear( self.num_wide, self.num_wide), nn.ReLU(),
        #     *hiddens
        # )
        # self.linears2_2 = nn.Sequential(
        #     nn.Linear((self.num_wide) if self.short_cut2 else self.num_wide, self.num_wide), nn.ReLU(),
        #     *hiddens
        # )
        # self.linears2_3 = nn.Sequential(
        #     nn.Linear(self.self.num_wide, self.self.num_wide), nn.ReLU(),
        #     nn.Linear(self.self.num_wide, self.self.num_wide), nn.ReLU(),
        # )
        # self.murho_activate = nn.Sequential(
        #     nn.Linear(self.self.num_wide, self.self.num_wide // 2), nn.ReLU(),
        #     nn.Linear(self.self.num_wide // 2, 5)
        # )
        # third MLP
        self.linears3_1 = nn.Sequential(
            nn.Linear( self.num_wide, self.num_wide), nn.ReLU(),
            *hiddens
        )
        # self.linears3_2 = nn.Sequential(
        #     nn.Linear((self.num_wide), self.num_wide), nn.ReLU(),
        #     *hiddens
        # )
        self.w_activate = nn.Sequential(
            nn.Linear(self.num_wide, self.num_wide*2), nn.ReLU(),
            nn.Linear(self.num_wide*2, output_size)
        )

        self.linears1_1.apply(self.initial_weights)
        # self.linears1_2.apply(self.initial_weights)
        # self.linears1_3.apply(self.initial_weights)
        self.sigma_activate.apply(self.initial_weights)
        # self.linears2_1.apply(self.initial_weights)
        # self.linears2_2.apply(self.initial_weights)
        # self.linears2_3.apply(self.initial_weights)
        # self.murho_activate.apply(self.initial_weights)
        self.linears3_1.apply(self.initial_weights)
        # self.linears3_2.apply(self.initial_weights)
        self.w_activate.apply(self.initial_weights)



    def forward(self,x):
        # forward
        distance = abs(x[:,0]-x[:,1])
        x = self.linears1_1(x)
        # x = self.linears1_2(x)
        # x = self.linears1_3(x)
        sigma = self.sigma_activate(x)
        sigma1, sigma2 =  100*distance* torch.sigmoid(sigma[:, 0]) + 0.0001, 100*distance * torch.sigmoid(sigma[:, 1]) + 0.0001 #为了减小sigma的范围，否则好像输出几千。可能是为了数值稳定性吧
        sigma = torch.stack([sigma1, sigma2], dim=1)
        # print(sigma)
        tmp_g1 = gaussian_layer(self.args.cal_size, sigma, kernel_size=self.args.k_size)

        # original input: (B*k*k),2
        # input: i) CoCs of RoI. Size:B,k*k,2. 2) pixel value RGB. Size: B*k*k,3
        # Output: target pixel value of focused image. Size:(B,3)
        # imgB1 = self.gaussian_new()

        # assert not torch.any(torch.isnan(imgB0))


        # x = self.linears2_1(x)
        # x = self.linears2_2(x)
        # x = self.linears2_3(x)
        # murho = self.murho_activate(x)
        # sigma1, sigma2 = 5 * torch.sigmoid(murho[:, 0]) + 0.0001, 5 * torch.sigmoid(murho[:, 1]) + 0.0001
        # mu1, mu2 = murho[:, 2] + 0.0001, murho[:, 3] + 0.0001
        # rho = torch.sigmoid(murho[:, 4]) * 0.1
        # sigma = torch.stack([sigma1, sigma2], dim=1)
        # mu = torch.stack([mu1, mu2], dim=1)
        # tmp_g2 = gaussian_layer(AIF_ROI, sigma, mu, rho)

        # assert not torch.any(torch.isnan(imgB))
        # print(imgC[0],imgB[0])
        # time5 = time.time()
        x = self.linears3_1(x)
        # x = self.linears3_2(x)
        x = self.w_activate(x)



        # gauss_paras = [sigma1, sigma2, mu1, mu2, rho]
        # gauss_paras = torch.stack(gauss_paras, 1)

        return tmp_g1,x



    def initial_weights(self,m):
        if isinstance(m, nn.Linear):
            if m.weight.shape[0] in [2, 3]:
                nn.init.xavier_normal_(m.weight, 0.1)
            else:
                nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)



def get_embedder(multires, i=-1, input_dim=3): # TisaiYu[2024/1/15] 位置编码，暂时没用，因为输入最小元素是每个像素点，感觉没法位置编码。
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = PositionEmbedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


