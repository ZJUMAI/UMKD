import os
import sys
sys.path.append("/data4/tongshuo/Grading/CommonFeatureLearning/models")
import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.soft_target_and_lossfunc import l2_loss
# from losses import ssim
import pvt

l2_loss=nn.MSELoss()

def to_pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, net):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.net = net

    def forward(self, x, **kwargs):
        return self.net(self.norm(x), **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_per_head ** -0.5

        inner_dim = dim_per_head * num_heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.attend = nn.Softmax(dim=-1)

        project_out = not (num_heads == 1 and dim_per_head == dim)
        self.out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, l, d = x.shape

        '''i. QKV projection'''
        # (b,l,dim_all_heads x 3)
        qkv = self.to_qkv(x)
        # (3,b,num_heads,l,dim_per_head)
        qkv = qkv.view(b, l, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        # 3 x (1,b,num_heads,l,dim_per_head)
        q, k, v = qkv.chunk(3)
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        '''ii. Attention computation'''
        attn = self.attend(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale
        )

        '''iii. Put attention on Value & reshape'''
        # (b,num_heads,l,dim_per_head)
        z = torch.matmul(attn, v)
        # (b,num_heads,l,dim_per_head)->(b,l,num_heads,dim_per_head)->(b,l,dim_all_heads)
        z = z.transpose(1, 2).reshape(b, l, -1)
        # assert z.size(-1) == q.size(-1) * self.num_heads

        '''iv. Project out'''
        # (b,l,dim_all_heads)->(b,l,dim)
        out = self.out(z)
        # assert out.size(-1) == d

        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, mlp_dim, depth=6, num_heads=8, dim_per_head=64, dropout=0.):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, num_heads=num_heads, dim_per_head=dim_per_head, dropout=dropout)),
                PreNorm(dim, FFN(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for norm_attn, norm_ffn in self.layers:
            x = x + norm_attn(x)
            x = x + norm_ffn(x)

        return x


class EnDe(nn.Module):
    def __init__(self):
        super(EnDe, self).__init__()
        # out_channels = [2 ** (i + 6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.d = pvt.feature_pvt_v2_b3()
        ckpt = torch.load('/data4/tongshuo/Grading/Ord2Seq/models/pvt_v2_b3.pth')
        self.d.load_state_dict(ckpt)

        # self.dhead = nn.Linear(64, args.num_classes)
        # self.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])
        # self.dhead = nn.Linear(512, self.output_dims + 1)


    def forward(self, x):
        b, c, h, w = x.shape
        # x = torch.cat([x, y], dim=1)
        out = self.d(x)  #图像编码器
        # out = out.mean(dim=0).squeeze(dim=1)  #(49 (downsample后的图像 7*7), batch_size, 512 (通道数) ) -> (batch_size, 512)
        # out = self.dhead(out)  #dhead 是分类头Linear， 512->num_class
        out = out.permute(1, 2, 0).unsqueeze(-1)
        # print('out:', out.shape)
        return out
        

class PVT_Linear(nn.Module):
    def __init__(self, num_classes):
        super(PVT_Linear, self).__init__()
        self.feature = EnDe()
        # self.criterion = nn.CrossEntropyLoss(reduction='none')
        #self.emb_dims = 256
        self.emb_dims = 512
        self.num_classes = num_classes
        #self.proj_conv = nn.Conv2d(512, self.emb_dims, 1)
        self.pool_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)), #outputs (bs, ps,1,1)
            nn.Flatten() #outputs (bs, ps)
            )
        # self.classifier = nn.Linear(self.emb_dims, args.num_classes)   # 线性分类器
        self.name = 'PVT_Linear'
        print("Use PVT with Linear")
        # classification head
        self.classifier = nn.Linear(self.emb_dims, num_classes) if num_classes > 0 else nn.Identity()
    def model_name(self):
        return self.name

    #def forward(self, x, target_x):
    def forward(self, x):
        # print("model input", x.shape, y.shape, target_x.shape, target_y.shape )
        # b = x.shape[0]
        x_emb = self.feature(x)
        #x_emb = x_emb.permute(1,0,2)
        #print('x_emb:',x_emb.shape)
        #x_emb = self.proj_conv(x_emb)
        x_pool = self.pool_layer(x_emb)
        #x_pool = self.pool_layer(x_emb)
        # print('x_pool:', x_pool.shape)
        x_cls = self.classifier(x_pool)
        # print(x_cls.shape)
        logits = x_cls
        # print(logits.shape)
        # loss_x_input = self.criterion(logits, target_x)
        # return logits, loss_x_input.mean()
        return logits
    #def val_inference(self, x, target_x):
    def val_inference(self, x):
        # print("model input", x.shape, y.shape, target_x.shape, target_y.shape )
        b = x.shape[0]
        x_emb = self.net(x)
        x_emb = self.proj_conv(x_emb)
        x_pool = self.pool_layer(x_emb)
        x_cls = self.classifier(x_pool)
        # print(x_cls.shape)
        logits = x_cls
        # print(logits.shape)
        
        # loss_x_input = self.criterion(logits, target_x)
        # return logits, loss_x_input.mean()
        return logits
    #def infernece(self, x, target_x):
    def infernece(self, x):
        b = x.shape[0]
        x_cls = self.net(x)
        x_cls = self.classifier(x_cls)
        
        return x_cls
    

class Args():
    def __init__(self):
        #self.no_GAN = True
        #self.weights = [2, 5, 0.1]
        self.num_classes = 5


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    args = Args()
    data = torch.rand([2, 3, 224, 224]).cuda()
    label = torch.tensor([1, 2]).cuda()
    # net = PVT_Linear(args).cuda()
    net = PVT_Linear(num_classes=5).cuda()
    print('in_features:', net.classifier.in_features)

    def forward_hook(module, input, output):
        module.output = output # keep feature maps
    net.feature.register_forward_hook(forward_hook)
    for name, param in net.named_parameters():
        if 'classifier' in name:
            print(name, param.shape)
        
    pred, loss = net(data)
    ft1 = net.feature.output
    print('pred:', pred)
    print('loss:', loss)
    print('feature:', ft1.shape)
    # pred = net(data, data, label, label)
    # pred = net(data, data, label, label, GAN=True)
    # print(pred.shape)



