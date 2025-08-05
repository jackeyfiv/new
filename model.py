import os

from torch.utils.flop_counter import FlopCounterMode

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
#from fvcore.nn import FlopCountAnalysis ,
# --------------- Encoder --------------------------
#多层感知机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)
#初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#超像素模块
class SP(nn.Module):
    def __init__(self, embed_dim, spixel_height, spixel_width):
        super().__init__()
        self.spixel_height = spixel_height
        self.spixel_width = spixel_width

        self.produce_q = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.LeakyReLU(0.1))
        self.softmax = nn.Softmax(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.size()
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        Q_prob = self.produce_q(x)
        Q_prob = self.softmax(Q_prob)

        prob_hot = Q_prob.clone()
        assig_max, _ = torch.max(prob_hot, dim=1, keepdim=True)
        prob_hot = torch.where(prob_hot == assig_max, torch.ones(prob_hot.shape).cuda(), torch.zeros(prob_hot.shape).cuda())

        x = self.feat_SP_transform(x, prob_hot, self.spixel_height, self.spixel_width)

        x = x.flatten(2).transpose(1, 2)

        return x, Q_prob
#超像素特征转换
    def feat_SP_transform(self, input, prob, spixel_height, spixel_width):
        def feat_prob_sum(feat_sum, shift_feat):
            feat_sum += shift_feat
            return feat_sum

        b, _, h, w = input.shape

        h_shift_unit = spixel_height
        w_shift_unit = spixel_width
        p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)  # prob: torch.Size([2, 9, 24, 24])
        feat_ = input  # b* (n+1) *h*w         feat_shape:  torch.Size([2, 4+1, 384, 384])
        # prob narrow shape: torch.Size([2, 1, 384, 384]) # test = feat_ * prob.narrow(1, 0, 1)   # print('test.shape', test.shape) test.shape torch.Size([2, 5, 384, 384])
        prob_feat = feat_ * prob.narrow(1, 0, 1)  # prob_feat shape:  torch.Size([2, 5, 24, 24])  sp_h 16
        # prob shape torch.Size([2, 9, 384, 384])   # prob.narrow(1, 0, 1) shape   torch.Size([2, 1, 384, 384])
        temp = F.pad(prob_feat, p2d, mode='constant', value=0)  # temp shape torch.Size([2, 5, 26, 26])
        send_to_top_left = temp[:, :, 2 * h_shift_unit:,
                           2 * w_shift_unit:]  # send_to_top_left.shape torch.Size([2, 5, 24, 24])
        feat_sum = send_to_top_left.clone()  # feat_sum.shape torch.Size([2, 4, 24, 24])

        prob_feat = feat_ * prob.narrow(1, 1, 1)  # b * (n+1) * h* w
        top = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum,
                                 top)  # prob_sum.shape torch.Size([2, 1, 24, 24])  prob_sum.shape torch.Size([2, 1, 24, 24])

        prob_feat = feat_ * prob.narrow(1, 2, 1)  # b * (n+1) * h* w
        top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, 2 * h_shift_unit:, :-2 * w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum, top_right)

        prob_feat = feat_ * prob.narrow(1, 3, 1)  # b * (n+1) * h* w
        left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
        feat_sum = feat_prob_sum(feat_sum, left)

        prob_feat = feat_ * prob.narrow(1, 4, 1)  # b * (n+1) * h* w
        center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit,
                 w_shift_unit:-w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum, center)

        prob_feat = feat_ * prob.narrow(1, 5, 1)  # b * (n+1) * h* w
        right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum, right)

        prob_feat = feat_ * prob.narrow(1, 6, 1)  # b * (n+1) * h* w
        bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, 2 * w_shift_unit:]
        feat_sum = feat_prob_sum(feat_sum, bottom_left)

        prob_feat = feat_ * prob.narrow(1, 7, 1)  # b * (n+1) * h* w
        bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum, bottom)

        prob_feat = feat_ * prob.narrow(1, 8, 1)  # b * (n+1) * h* w
        bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
        feat_sum = feat_prob_sum(feat_sum, bottom_right)

        return feat_sum
#可分离注意力
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_mlp=True):
        super().__init__()
        self.is_mlp = is_mlp
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # self.sp = SP(embed_dim=dim, spixel_height=spixel_height, spixel_width=spixel_width)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.attn(self.norm1(x), H, W)
        # x, sp_Q = self.sp(x, H, W)
        x = x + self.drop_path(x)
        if self.is_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], spixel_height=[16,8,8,4], spixel_width=[16,8,8,4]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 2, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.sp1 = SP(embed_dim=embed_dims[0], spixel_height=spixel_height[0], spixel_width=spixel_width[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.sp2 = SP(embed_dim=embed_dims[1], spixel_height=spixel_height[1], spixel_width=spixel_width[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.sp3 = SP(embed_dim=embed_dims[2], spixel_height=spixel_height[2], spixel_width=spixel_width[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.sp4 = SP(embed_dim=embed_dims[3], spixel_height=spixel_height[3], spixel_width=spixel_width[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        Q_prob_collect = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x, Q_prob = self.sp1(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        Q_prob_collect.append(Q_prob)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x, Q_prob = self.sp2(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        Q_prob_collect.append(Q_prob)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x, Q_prob = self.sp3(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        Q_prob_collect.append(Q_prob)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x, Q_prob = self.sp4(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        Q_prob_collect.append(Q_prob)

        return outs, Q_prob_collect

    def forward(self, x):
        x, Q_prob_collect = self.forward_features(x)
        # x = self.head(x)
        # print('encoder shape: ', x[3].shape)
        return x, Q_prob_collect


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

# --------------- Decoder --------------------------
# class Dim_down(nn.Module):
#     def __init__(self, input_dims, output_dims):
#         super().__init__()
#         self.dim_down = nn.Conv2d(input_dims, output_dims, 1, 1, 0)
#         self.norm = nn.LayerNorm(output_dims)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         x = self.dim_down(x)
#         _, _, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)

        # return x, H, W

class Dim_down(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.dim_down = nn.Conv2d(input_dims, output_dims, 1, 1, 0)
        self.norm = nn.BatchNorm2d(output_dims)
        self.act_layer = nn.GELU()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.dim_down(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = self.act_layer(x)
        x = x.flatten(2).transpose(1, 2)

        return x, H, W

class SegFormerHead(nn.Module):
    # def __init__(self, in_channels=[64,128,320,512], feature_strides=[4, 8, 16, 32], decoder_params=256, num_classes=19):
    def __init__(self, drop_path_rate=0., drop_rate=0., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 attn_drop_rate=0., embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1], num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_dim_down = Dim_down(embed_dims[3]+embed_dims[2], embed_dims[2])

        self.upup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upup_dim_down = Dim_down(embed_dims[2] + embed_dims[1], embed_dims[1])

        self.upupup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upupup_dim_down = Dim_down(embed_dims[1] + embed_dims[0], embed_dims[0])

        self.upupupup = nn.ConvTranspose2d(embed_dims[0], embed_dims[0] // 2, kernel_size=2, stride=2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], is_mlp=True)
            for i in range(depths[2])])
        self.norm1 = norm_layer(embed_dims[2])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], is_mlp=True)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], is_mlp=True)
            for i in range(depths[0])])
        self.norm3 = norm_layer(embed_dims[0])

        # cur += depths[2]
        # self.block4 = nn.ModuleList([Block(
        #     dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[3])
        #     for i in range(depths[3])])
        # self.norm4 = norm_layer(embed_dims[3])

        self.fuse_norm = nn.BatchNorm2d(embed_dims[0] // 2)
        self.pred_output = nn.Conv2d(embed_dims[0] // 2, self.num_classes, kernel_size=1)
        self.pred_norm = nn.BatchNorm2d(self.num_classes)

    def forward(self, fm):
        f, ff, fff, ffff = fm
        B = f.shape[0]
        # stage 1
        ffff_u = self.up(ffff)
        x = torch.cat([fff,ffff_u], dim=1) # torch.Size([1, 832, 14, 14])
        x, H, W = self.up_dim_down(x)   # torch.Size([1, 320, 14, 14])
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
            x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([1, 320, 14, 14])

        # stage 2
        x_u = self.upup(x)
        x = torch.cat([ff, x_u], dim=1)
        x, H, W = self.upup_dim_down(x)         # torch.Size([1, 784, 128]) 28 28
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
            x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([1, 128, 28, 28]) 28 28

        # stage 3
        x_u = self.upupup(x)
        x = torch.cat([f,x_u], dim=1)
        x, H, W = self.upupup_dim_down(x)   # torch.Size([1, 3136, 64]) 56 56
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
            x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # torch.Size([1, 64, 56, 56]) 56 56
        # print('x.shape', x.shape, H, W)

        # stage 4
        x = self.upupupup(x)

        x = self.fuse_norm(x)
        # output
        x = self.pred_output(x)
        x = self.pred_norm(x)


        return x

#--------------Encoder + Decoder -----------------------------

class Stage_SSM(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.encoder = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=nn.LayerNorm, depths=[2, 2, 2, 2], sr_ratios=[1, 1, 1, 1],
            drop_rate=0.0, drop_path_rate=0, spixel_height=[2,2,2,2], spixel_width=[2,2,2,2])
        self.decoder = SegFormerHead(drop_path_rate=0., drop_rate=0., qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm,
                 attn_drop_rate=0., embed_dims=[64, 128, 320, 512], depths=[1, 1, 1, 1], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[1, 1, 1, 1], num_classes=num_class)

    def forward(self, x):
        x, Q_prob_collect = self.encoder(x)
        # print('x[0].shape, x[1].shape, x[2].shape, x[3].shape', x[0].shape, x[1].shape, x[2].shape, x[3].shape)
        # x[0].shape, x[1].shape, x[2].shape, x[3].shape torch.Size([1, 64, 56, 56]) torch.Size([1, 128, 28, 28]) torch.Size([1, 320, 14, 14]) torch.Size([1, 512, 7, 7])
        # x[0].shape, x[1].shape, x[2].shape, x[3].shape torch.Size([1, 64, 32, 32]) torch.Size([1, 128, 16, 16]) torch.Size([1, 320, 8, 8]) torch.Size([1, 512, 4, 4])
        x = self.decoder(x)
        return x, Q_prob_collect


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")  # use CPU or GPU

    # x = torch.randn(2,1,224,224)
    x = torch.randn(1, 1, 256, 256)
    x = x.to(device)
    model = Stage_SSM(4)
    print('model parameters:',sum(p.numel() for p in model.parameters())/1e6,'M')
    model = model.to(device)
    model = nn.DataParallel(model)
    y, Q_prob_collect = model(x)
    #print(y.shape)
    #print(Q_prob_collect[0].shape, Q_prob_collect[1].shape, Q_prob_collect[2].shape, Q_prob_collect[3].shape)

    # torch.Size([2, 9, 32, 32]) torch.Size([2, 9, 16, 16]) torch.Size([2, 9, 8, 8]) torch.Size([2, 9, 4, 4]) for 128x128 image
