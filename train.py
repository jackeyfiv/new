import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append('comparison_methods')
import torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
# from utils.dice_score import dice_loss
import torch.nn as nn
import os
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
print('GPU num: ', torch.cuda.device_count())
from torch.utils.data import DataLoader
from torch import optim
from Stage_SSM import Stage_SSM
from dataloading import BasicDataset, CarvanaDataset
from utils.dice_score import multiclass_dice_coeff, dice_coeff
def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor
#初始化像素矩阵
def init_spixel_grid(img_height, img_width, batch_size, device=None):
    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
    
    all_XY_feat = torch.from_numpy(
        np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)
    )
    
    if device is not None:
        all_XY_feat = all_XY_feat.to(device)
        
    return all_XY_feat

def build_LABXY_feat(label_in, XY_feat):
    img_lab = label_in.clone().type(torch.float)
    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img =  F.interpolate(img_lab, size=(curr_img_height,curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat],dim=1)

    return LABXY_feat


def dice_loss_multiclass(pred, target, smooth=1e-5):
    """
    多类Dice损失函数
    Args:
        pred: 模型预测的概率分布 (B, C, H, W)
        target: 真实标签 (B, H, W) 值为0,1,2
        smooth: 平滑系数避免除零
    Returns:
        dice_loss: 多类Dice损失值
    """
    # 将真实标签转换为one-hot编码 (B, C, H, W)
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    # 计算交集和并集
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    # 计算每个类别的Dice系数
    dice = (2. * intersection + smooth) / (union + smooth)

    # 返回平均Dice损失 (1 - Dice系数)
    return 1 - dice.mean()


# 多类Dice评估指标
def dice_metric_multiclass(pred, target, smooth=1e-5):
    """
    多类Dice评估指标
    Args:
        pred: 模型预测的类别 (B, H, W) 值为0,1,2
        target: 真实标签 (B, H, W) 值为0,1,2
        smooth: 平滑系数避免除零
    Returns:
        dice_score: 多类Dice分数
    """
    # 将预测和标签转换为one-hot编码
    pred_onehot = F.one_hot(pred, num_classes=3).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    # 计算交集和并集
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    # 计算每个类别的Dice系数
    dice_per_class = (2. * intersection + smooth) / (union + smooth)

    # 返回各类别的平均Dice分数
    return dice_per_class.mean(dim=0)  # 返回每个类别的Dice分数


def dice_loss_multiclass(pred, target, smooth=1e-5):
    """
    多类Dice损失函数
    Args:
        pred: 模型预测的概率分布 (B, C, H, W)
        target: 真实标签 (B, H, W) 值为0,1,2
        smooth: 平滑系数避免除零
    Returns:
        dice_loss: 多类Dice损失值
    """
    # 将真实标签转换为one-hot编码 (B, C, H, W)
    target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    # 计算交集和并集
    intersection = (pred * target_onehot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

    # 计算每个类别的Dice系数
    dice = (2. * intersection + smooth) / (union + smooth)

    # 返回平均Dice损失 (1 - Dice系数)
    return 1 - dice.mean()
def poolfeat(input, prob, sp_h=2, sp_w=2):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)                          #prob: torch.Size([2, 9, 24, 24])
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w         feat_shape:  torch.Size([2, 4+1, 384, 384])
    # prob narrow shape: torch.Size([2, 1, 384, 384]) # test = feat_ * prob.narrow(1, 0, 1)   # print('test.shape', test.shape) test.shape torch.Size([2, 5, 384, 384])
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) #prob_feat shape:  torch.Size([2, 5, 24, 24])  sp_h 16
    # prob shape torch.Size([2, 9, 384, 384])   # prob.narrow(1, 0, 1) shape   torch.Size([2, 1, 384, 384])
    temp = F.pad(prob_feat, p2d, mode='constant', value=0)   # temp shape torch.Size([2, 5, 26, 26])
    send_to_top_left = temp[:, :, 2 * h_shift_unit:, 2 * w_shift_unit:]  # send_to_top_left.shape torch.Size([2, 5, 24, 24])
    feat_sum = send_to_top_left[:, :-1, :, :].clone()  # feat_sum.shape torch.Size([2, 4, 24, 24])
    prob_sum = send_to_top_left[:, -1:, :, :].clone()  # prob_sum.shape torch.Size([2, 1, 24, 24])

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top) # prob_sum.shape torch.Size([2, 1, 24, 24])  prob_sum.shape torch.Size([2, 1, 24, 24])

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)


    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat
def upfeat(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # todo: currently we assume the downsize scale in x,y direction are always same

    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape   #torch.Size([8, 52, 384, 384])
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)  #  pooled_labxy.shape   torch.Size([2, 52, 24, 24])

    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)  # reconstr_feat.shape torch.Size([2, 52, 384, 384])

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]   #loss_map.shape torch.Size([2, 2, 384, 384])

    # self def cross entropy  -- the official one combined softmax
    logit = torch.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - torch.sum(logit * labxy_feat[:, :-2, :, :]) / b
    loss_pos = torch.norm(loss_map, p=2, dim=1).sum() / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum = 0.005 * (loss_sem + loss_pos)
    loss_sem_sum = 0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def custom_collate_fn(batch):
    valid_batch = []
    for item in batch:
        if isinstance(item, dict) and 'image' in item and 'mask' in item:
            # 确保mask有通道维度 [1, H, W]
            mask = item['mask']
            if mask.dim() == 2:  # 如果是2D [H, W]
                mask = mask.unsqueeze(0)  # 添加通道维度 -> [1, H, W]
            valid_batch.append({'image': item['image'], 'mask': mask})

    if not valid_batch:
        raise RuntimeError("No valid items in batch!")

    images = torch.stack([item['image'] for item in valid_batch])
    masks = torch.stack([item['mask'] for item in valid_batch])

    return {'image': images, 'mask': masks}

def dice_loss_binary(pred, target, smooth=1e-5):
    """
    二类Dice损失函数
    Args:
        pred: 模型预测的概率分布 (B, 1, H, W)
        target: 真实标签 (B, H, W) 值为0或1
        smooth: 平滑系数避免除零
    Returns:
        dice_loss: 二类Dice损失值
    """
    pred = torch.sigmoid(pred)  # 确保在0-1范围内
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def dice_metric_binary(pred, target, smooth=1e-5):
    """
    修正二分类Dice计算
    Args:
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
    """
    # 展平成向量计算
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    total = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice


def train_model_binary(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, total_epoch, model_name,
                       state_save_path, state_load_path=None):
    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))
    
    best_val_dice = 0.0
    num_step = len(train_loader)

    for epoch in range(num_epochs):
        # ================= 训练阶段 =================
        model.train()
        epoch_loss = 0
        train_dice_score = 0.0

        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).float()
                batch_step = images.shape[0]

                # 动态创建XY_feat，使用当前批次大小
                xy_feat1 = init_spixel_grid(64, 64, batch_step, device=device)
                xy_feat2 = init_spixel_grid(32, 32, batch_step, device=device)
                xy_feat3 = init_spixel_grid(16, 16, batch_step, device=device)
                xy_feat4 = init_spixel_grid(8, 8, batch_step, device=device)

                masks1 = F.interpolate(masks, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)
                
                # 二分类只需要一个通道的输出 - 修正这里
                masks_pred = masks_pred[:, :1]  # 只取第一个通道 [B, 1, H, W]
                
                # 调整特征图大小
                LABXY_feat_tensor1 = F.interpolate(LABXY_feat_tensor1, size=Q_prob_collect[0].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor2 = F.interpolate(LABXY_feat_tensor2, size=Q_prob_collect[1].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor3 = F.interpolate(LABXY_feat_tensor3, size=Q_prob_collect[2].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor4 = F.interpolate(LABXY_feat_tensor4, size=Q_prob_collect[3].shape[2:],
                                                   mode='bilinear', align_corners=False)

                slic_loss1, _, _ = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss2, _, _ = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss3, _, _ = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3,
                                                             pos_weight=0.003, kernel_size=2)
                slic_loss4, _, _ = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4,
                                                             pos_weight=0.003, kernel_size=1)

                # 二分类损失计算 - 修正这里
                loss_value = criterion(masks_pred, masks)  # 使用BCEWithLogitsLoss
                loss_dice = dice_loss_binary(torch.sigmoid(masks_pred), masks)
                
                loss_sum = loss_value + loss_dice + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                # 二分类预测
                pred_classes = (torch.sigmoid(masks_pred) > 0.5).float()
                batch_dice = dice_metric_binary(pred_classes, masks)
                train_dice_score += batch_dice.item()

                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'dice': train_dice_score / (iteration + 1)})
                pbar.update(1)

        # ================= 验证阶段 =================
        model.eval()
        val_dice_score = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch['image'].to(device)
                val_masks = val_batch['mask'].to(device).float()
                
                val_preds, _ = model(val_images)
                val_preds = val_preds[:, :1]  # 只取第一个通道 [B, 1, H, W]
                
                # 二分类预测
                val_pred_classes = (torch.sigmoid(val_preds) > 0.5).float()
                val_dice = dice_metric_binary(val_pred_classes, val_masks)
                
                val_dice_score += val_dice.item() * val_images.size(0)
                val_samples += val_images.size(0)
        
        avg_val_dice = val_dice_score / val_samples
        print(f"Epoch {epoch+1} Validation Dice: {avg_val_dice:.4f}")
        
        # 保存最佳模型
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), f"{state_save_path}{model_name}_best.pth")
            print(f"Saved best model with Dice: {best_val_dice:.4f}")

        scheduler.step()

    # 保存最终模型
    torch.save(model.state_dict(), f"{state_save_path}{model_name}_final.pth")
def train_model1(model, criterion, optimizer, scheduler, train_loader, num_epochs, total_epoch, model_name,
                 state_save_path, state_load_path=None):
    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)

    def init_spixel_grid(img_height, img_width, batch_size):
        curr_img_height = int(np.floor(img_height))
        curr_img_width = int(np.floor(img_width))

        all_h_coords = np.arange(0, curr_img_height, 1)
        all_w_coords = np.arange(0, curr_img_width, 1)
        curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
        coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])
        all_XY_feat = (torch.from_numpy(np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)))
        return all_XY_feat.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dice_score = 0.0  # 改为浮点数

        xy_feat1 = init_spixel_grid(64, 64, batch_size)
        xy_feat2 = init_spixel_grid(32, 32, batch_size)
        xy_feat3 = init_spixel_grid(16, 16, batch_size)
        xy_feat4 = init_spixel_grid(8, 8, batch_size)

        model.train()
        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device).float()
                batch_step = images.shape[0]

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(64, 64, batch_step)
                    xy_feat2 = init_spixel_grid(32, 32, batch_step)
                    xy_feat3 = init_spixel_grid(16, 16, batch_step)
                    xy_feat4 = init_spixel_grid(8, 8, batch_step)

                masks1 = F.interpolate(masks, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                LABXY_feat_tensor1 = F.interpolate(LABXY_feat_tensor1, size=Q_prob_collect[0].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor2 = F.interpolate(LABXY_feat_tensor2, size=Q_prob_collect[1].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor3 = F.interpolate(LABXY_feat_tensor3, size=Q_prob_collect[2].shape[2:],
                                                   mode='bilinear', align_corners=False)
                LABXY_feat_tensor4 = F.interpolate(LABXY_feat_tensor4, size=Q_prob_collect[3].shape[2:],
                                                   mode='bilinear', align_corners=False)

                slic_loss1, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1,
                                                                           pos_weight=0.003, kernel_size=2)
                slic_loss2, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2,
                                                                           pos_weight=0.003, kernel_size=2)
                slic_loss3, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3,
                                                                           pos_weight=0.003, kernel_size=2)
                slic_loss4, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4,
                                                                           pos_weight=0.003, kernel_size=1)

                masks_pred = F.softmax(masks_pred, dim=1)

                loss_value = criterion(masks_pred, masks.long().squeeze(1)) + \
                             dice_loss_binary(masks_pred[:, 1:2, :, :], masks)  # 只取前景通道
                loss_sum = loss_value + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                masks_pred = (masks_pred > 0.5).float()
                pred_classes = torch.argmax(masks_pred, dim=1)

                # 修改这里：计算平均Dice分数
                #
                pred_classes = (torch.sigmoid(masks_pred[:, 1]) > 0.5).float()  # 取前景概率>0.5为预测
                dice_score += dice_metric_binary(pred_classes, masks.squeeze(1)).item()

                # 更新进度条
                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'dice': dice_score / (iteration + 1)})
                pbar.update(1)

        scheduler.step()

    torch.save(model.state_dict(), state_save_path + model_name + '_{}.pth'.format(total_epoch))
def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, total_epoch, model_name,
                state_save_path, state_load_path=None):
    #     torch.backends.cudnn.benchmark = True

    if state_load_path is not None:
        model.load_state_dict(torch.load(state_load_path))

    num_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        dice_score = 0

        xy_feat1 = init_spixel_grid(64, 64, batch_size)
        xy_feat2 = init_spixel_grid(32, 32, batch_size)
        xy_feat3 = init_spixel_grid(16, 16, batch_size)
        xy_feat4 = init_spixel_grid(8, 8, batch_size)

        model.train()
        with tqdm(total=num_step, desc=f'Epoch {epoch + 1}/{num_epochs}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                #images, masks = batch[0], batch[1]
                images = batch['image'] # 提取批次中的第一个样本
                masks = batch['mask']
                batch_step = images.shape[0]
                images = images.to(device)
                masks = masks.to(device)

                if iteration == num_step - 1:
                    xy_feat1 = init_spixel_grid(64, 64, batch_step)
                    xy_feat2 = init_spixel_grid(32, 32, batch_step)
                    xy_feat3 = init_spixel_grid(16, 16, batch_step)
                    xy_feat4 = init_spixel_grid(8, 8, batch_step)

                masks1 = F.interpolate(masks, size=(64, 64), mode='nearest')
                masks2 = F.interpolate(masks, size=(32, 32), mode='nearest')
                masks3 = F.interpolate(masks, size=(16, 16), mode='nearest')
                masks4 = F.interpolate(masks, size=(8, 8), mode='nearest')

                LABXY_feat_tensor1 = build_LABXY_feat(masks1, xy_feat1)
                LABXY_feat_tensor2 = build_LABXY_feat(masks2, xy_feat2)
                LABXY_feat_tensor3 = build_LABXY_feat(masks3, xy_feat3)
                LABXY_feat_tensor4 = build_LABXY_feat(masks4, xy_feat4)

                masks_pred, Q_prob_collect = model(images)

                slic_loss1, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[0], LABXY_feat_tensor1, pos_weight=0.003, kernel_size=2)
                slic_loss2, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[1], LABXY_feat_tensor2, pos_weight=0.003, kernel_size=2)
                slic_loss3, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[2], LABXY_feat_tensor3, pos_weight=0.003, kernel_size=2)
                slic_loss4, loss_sem, loss_pos = compute_semantic_pos_loss(Q_prob_collect[3], LABXY_feat_tensor4, pos_weight=0.003, kernel_size=1)


                masks_pred = F.softmax(masks_pred, dim=1)


                loss_value = criterion(masks_pred, masks.long().squeeze(1)) +  dice_loss_multiclass(masks_pred, masks.long().squeeze(1))
                loss_sum = loss_value + (slic_loss1 + slic_loss2 + slic_loss3 + slic_loss4) * 0.2
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                epoch_loss += loss_sum.item()

                masks_pred = (masks_pred > 0.5).float()
                pred_classes = torch.argmax(masks_pred,dim=1)
                dice_score += dice_metric_multiclass(pred_classes, masks.long().squeeze(1))

                pbar.set_postfix(**{'loss': epoch_loss / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'dice': dice_score.item() / (iteration + 1)})
                pbar.update(1)
        #             epoch_avg_loss = epoch_loss / sample_num

        scheduler.step()
    #         print('epoch_avg_loss: ', epoch_avg_loss)
    torch.save(model.state_dict(), state_save_path + model_name + '_{}.pth'.format(total_epoch))
if __name__ == '__main__':
    model = Stage_SSM(num_class=1)  # 从3改为2
    model = model.to(device)
    # model = nn.DataParallel(model)
    dir_img = Path(r'/content/gdrive/MyDrive/train/img')
    dir_mask = Path(r'/content/gdrive/MyDrive/train/mask')

    try:
        train_set = CarvanaDataset(dir_img, dir_mask, 1)
    except (AssertionError, RuntimeError):
        train_set = BasicDataset(dir_img, dir_mask, 1)
    dataset = BasicDataset(dir_img, dir_mask, 1)
    val_dir_img = Path('/content/gdrive/MyDrive/train/img')
    val_dir_mask = Path('/content/gdrive/MyDrive/train/mask')
    val_set = CarvanaDataset(val_dir_img, val_dir_mask, 1) or BasicDataset(val_dir_img, val_dir_mask, 1)
    
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {item['image'].shape}")
        print(f"  Mask shape: {item['mask'].shape}")
        print(f"  Unique mask values: {torch.unique(item['mask'])}")




    # 验证训练集和验证集
    # validate_dataset(train_set, "Training Set")
    # validate_dataset(val_set, "Validation Set")

    # 3. 创建数据加载器
    loader_args = dict(
        batch_size=16,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn  # 添加自定义collate函数
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    # 创建验证集加载器
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, 
                       num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)



    batch_size = 32
    learning_rate = 0.000075



    state_save_path = r'/content/gdrive/MyDrive/SSM1/.idea/checkpoint/'
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.91)
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    global_step = 0

############################################################################  以下是训练的相关设置代码


    # state_load_path = '/data/dwl/all_age_data/model/atrial_dataset_baseline_SP/Baseline-SP-atrial_50.pth'
    train_model_binary(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    scheduler=lr_scheduler,
    train_loader=train_loader,
    val_loader=val_loader,  # 添加验证集
    num_epochs=50, 
    total_epoch=50, 
    model_name='Stage_SSM_binary',
    state_save_path=state_save_path
)

