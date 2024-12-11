import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )

        self.pooling=nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b,_,c=x[0].shape
        x1=self.pooling(rearrange(x[0],'b n c -> b c n')).view(b,c)
        x2=self.pooling(rearrange(x[1],'b n c -> b c n')).view(b,c)
        x3=self.pooling(rearrange(x[2],'b n c -> b c n')).view(b,c)
        scores=self.net(torch.cat((x1,x2,x3),dim=-1))
        scores=torch.chunk(scores,3,dim=-1)
        scores = [x.unsqueeze(1) for x in scores]

        x1=x[0]*scores[0].expand_as(x[0])+x[0]
        x2=x[1]*scores[1].expand_as(x[1])+x[1]
        x3=x[2]*scores[2].expand_as(x[2])+x[2]
        return [x1,x2,x3]
    
class SplitFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.split_mlp=nn.Sequential(
            nn.LayerNorm(dim//3),
            nn.Linear(dim//3, hidden_dim//3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//3, dim//3),
            nn.GELU(),
        )


    def forward(self, x):
        # cross-scale weight sharing
        x1=self.split_mlp(x[0])+x[0]
        x2=self.split_mlp(x[1])+x[1]
        x3=self.split_mlp(x[2])+x[2]

        return [x1,x2,x3]

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv1 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_qkv2 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_qkv3 = nn.Linear(dim//3, inner_dim, bias = False)

        self.to_out1 = nn.Sequential(
            nn.Linear(inner_dim//3, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim//3, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out3 = nn.Sequential(
            nn.Linear(inner_dim//3, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        qkv1=self.to_qkv1(x[0]).chunk(3, dim = -1)
        qkv2=self.to_qkv2(x[1]).chunk(3, dim = -1)
        qkv3=self.to_qkv3(x[2]).chunk(3, dim = -1)

        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads//3), qkv1)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots1)
        attn1 = self.dropout(attn1)

        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = self.to_out1(out1)+x[0]

        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads//3), qkv2)

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale

        attn2 = self.attend(dots2)
        attn2 = self.dropout(attn2)

        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_out2(out2)+x[1]

        q3, k3, v3 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads//3), qkv3)

        dots3 = torch.matmul(q3, k3.transpose(-1, -2)) * self.scale

        attn3 = self.attend(dots3)
        attn3 = self.dropout(attn3)

        out3 = torch.matmul(attn3, v3)
        out3 = rearrange(out3, 'b h n d -> b n (h d)')
        out3 = self.to_out3(out3)+x[2]

        return [out1,out2,out3]

class CrossScaleAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim//3)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q0 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_k0 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_v0 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_q1 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_k1 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_v1 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_q2 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_k2 = nn.Linear(dim//3, inner_dim, bias = False)
        self.to_v2 = nn.Linear(dim//3, inner_dim, bias = False)

        self.to_out0 = nn.Sequential(
            nn.Linear(2*inner_dim, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out1 = nn.Sequential(
            nn.Linear(2*inner_dim, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out2 = nn.Sequential(
            nn.Linear(2*inner_dim, dim//3),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x_n=[self.norm(t) for t in x]
        q0=self.to_q0(x_n[0])
        k0=self.to_k0(x_n[0])
        v0=self.to_v0(x_n[0])
        q1=self.to_q1(x_n[1])
        k1=self.to_k1(x_n[1])
        v1=self.to_v1(x_n[1])
        q2=self.to_q2(x_n[2])
        k2=self.to_k2(x_n[2])
        v2=self.to_v2(x_n[2])

        q0,k0,v0,q1,k1,v1,q2,k2,v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q0,k0,v0,q1,k1,v1,q2,k2,v2))

        dots1 = torch.matmul(q0, k1.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots1)
        attn1 = self.dropout(attn1)

        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q0, k2.transpose(-1, -2)) * self.scale

        attn2 = self.attend(dots2)
        attn2 = self.dropout(attn2)

        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        outs=torch.cat((out1,out2),dim=-1)
        out0_final=self.to_out0(outs)

        #next
        dots0 = torch.matmul(q1, k0.transpose(-1, -2)) * self.scale

        attn0 = self.attend(dots0)
        attn0 = self.dropout(attn0)

        out0 = torch.matmul(attn0, v0)
        out0 = rearrange(out0, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q1, k2.transpose(-1, -2)) * self.scale

        attn2 = self.attend(dots2)
        attn2 = self.dropout(attn2)

        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        outs=torch.cat((out0,out2),dim=-1)
        out1_final=self.to_out1(outs)

        # next 
        dots1 = torch.matmul(q2, k1.transpose(-1, -2)) * self.scale

        attn1 = self.attend(dots1)
        attn1 = self.dropout(attn1)

        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots0 = torch.matmul(q2, k0.transpose(-1, -2)) * self.scale

        attn0 = self.attend(dots0)
        attn0 = self.dropout(attn0)

        out0 = torch.matmul(attn0, v0)
        out0 = rearrange(out0, 'b h n d -> b n (h d)')

        outs=torch.cat((out0,out1),dim=-1)
        out2_final=self.to_out2(outs)

        return [out0_final,out1_final,out2_final]
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim//3)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                SplitFeedForward(dim, mlp_dim, dropout = dropout),
                CrossScaleAttention(dim, heads = 3, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
            ]))

    def forward(self, x1, x2, x3):
        x = [x1,x2,x3]
        for attn, ff, attn2, ff2 in self.layers:
            x = attn(x)
            x = ff(x)
            x = attn2(x)
            x = ff2(x)
            
        return [self.norm(a) for a in x]

class M2OST(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=250, dim=768, depth=8, heads=12, mlp_dim=512, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_height % 8 == 0 and image_width % 8 == 0, 'Image size must be divisible by 8.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.GELU(),
            nn.Linear(dim,dim//3),
            nn.LayerNorm(dim//3),
        )
        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height//2, p2 = patch_width//2),
            nn.LayerNorm(patch_dim//4),
            nn.Linear(patch_dim//4, dim//2),
            nn.GELU(),
            nn.Linear(dim//2,dim//3),
            nn.LayerNorm(dim//3),
        )
        self.to_patch_embedding3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height//4, p2 = patch_width//4),
            nn.LayerNorm(patch_dim//16),
            nn.Linear(patch_dim//16, dim//4),
            nn.GELU(),
            nn.Linear(dim//4,dim//3),
            nn.LayerNorm(dim//3),
        )

        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches + 1, dim//3))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, 2*num_patches + 1, dim//3))
        self.pos_embedding3 = nn.Parameter(torch.randn(1, 3*num_patches + 1, dim//3))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim//3))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, img2, img3):
        
        # if img2 is None and img3 is None:
        #     _,_,image_height,image_width=img.shape
            
        #     height_left=image_height//4
        #     height_right=image_height-height_left
        #     width_left=image_width//4
        #     width_right=image_width-width_left
            
        #     height_midleft=3*image_height//8
        #     height_midright=image_height-height_left
        #     width_midleft=3*image_width//8
        #     width_midright=image_width-width_left

        #     img3=torch.nn.functional.interpolate(img,scale_factor=0.25,mode='bilinear')
        #     img2=torch.nn.functional.interpolate(img[:,:,height_left:height_right,width_left:width_right],scale_factor=0.5,mode='bilinear')
        #     img=img[:,:,height_midleft:height_midright,width_midleft:width_midright]
        
        _,_,image_height,image_width=img.shape
        
        x1 = self.to_patch_embedding1(img)

        height_left=image_height//4
        height_right=image_height-height_left
        width_left=image_width//4
        width_right=image_width-width_left
        
        x2_1=self.to_patch_embedding1(img2)
        img2_2=img2[:,:,height_left:height_right,width_left:width_right]
        x2_2 = self.to_patch_embedding2(img2_2)
        x2=torch.cat((x2_2,x2_1),dim=1)
        
        height_left=image_height//4
        height_right=image_height-height_left
        width_left=image_width//4
        width_right=image_width-width_left
        
        height_midleft=3*image_height//8
        height_midright=image_height-height_left
        width_midleft=3*image_width//8
        width_midright=image_width-width_left

        x3_1 = self.to_patch_embedding1(img3)
        img3_2=img3[:,:,height_left:height_right,width_left:width_right]
        x3_2 = self.to_patch_embedding2(img3_2)
        img3_3=img3[:,:,height_midleft:height_midright,width_midleft:width_midright]
        x3_3 = self.to_patch_embedding3(img3_3)
        x3=torch.cat((x3_3,x3_2,x3_1),dim=1)

        b, n1, _ = x1.shape
        b, n2, _ = x2.shape
        b, n3, _ = x3.shape

        cls_tokens1 = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x1 = torch.cat((cls_tokens1, x1), dim=1)
        x1 += self.pos_embedding1[:, :(n1 + 1)]
        x1 = self.dropout(x1)

        cls_tokens2 = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x2 = torch.cat((cls_tokens2, x2), dim=1)
        x2 += self.pos_embedding2[:, :(n2 + 1)]
        x2 = self.dropout(x2)

        cls_tokens3 = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x3 = torch.cat((cls_tokens3, x3), dim=1)
        x3 += self.pos_embedding3[:, :(n3 + 1)]
        x3 = self.dropout(x3)

        x = self.transformer(x1,x2,x3)

        x = [a.mean(dim = 1) for a in x] if self.pool == 'mean' else [a[:, 0] for a in x]
        x=torch.cat(x,dim=-1)
        return self.mlp_head(x)

if __name__=="__name__":
    inputs=torch.randn((1,3,224,224))
    model=M2OST(depth=6,dim=192*3, mlp_dim=192*2, heads=9,dim_head=64)
    # model.load_state_dict(torch.load('./',map_location='cpu'))
    out=model(inputs,inputs,inputs)