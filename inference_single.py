import numpy as np
import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk  
from easydict import EasyDict
from utils.config import get_config
from models.model_dict import get_model
from utils.data_us import JointTransform3D

def main():
    # ==================================================
    # 1. 接收大模型 Agent 传来的指令
    # ==================================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help="输入单图绝对路径 (.nii.gz)")
    parser.add_argument('--output_mask', type=str, required=True, help="输出掩码绝对路径 (.nii.gz)")
    parser.add_argument('--ckpt', type=str, required=True, help="模型权重绝对路径")
    cli_args = parser.parse_args()

    print(f"=== [MemSAM 单图推理引擎] 启动 ===")
    print(f"📥 正在读取: {cli_args.input_image}")

    # ==================================================
    # 2. 环境配置 
    # ==================================================
    args = EasyDict(
        base_lr=0.0005, batch_size=1, encoder_input_size=256, keep_log=False,
        low_image_size=256, modelname='MemSAM', n_gpu=1,
        sam_ckpt='pretrained/sam_vit_b_01ec64.pth', task='CAMUS_Video',
        vit_name='vit_b', enable_memory=True, disable_point_prompt=False,
        point_numbers_prompt=True, point_numbers=1, reinforce=False, compute_ef=False
    )
    
    opt = get_config(args.task)
    opt.mode = "test"
    opt.modelname = 'MemSAM'
    device = torch.device(opt.device if hasattr(opt, 'device') else 'cuda')

    # ==================================================
    # 3. 唤醒并加载模型 
    # ==================================================
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.eval() # 推理模式！

    checkpoint = torch.load(cli_args.ckpt, map_location=device)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    print("🧠 模型权重加载完毕！")
    # ==================================================
    # 4. 读取 .npy 视频流并抽取 5 帧
    # ==================================================
    img_array = np.load(cli_args.input_image) 
    print(f"  [数据处理] 原始视频张量维度: {img_array.shape}") # 预期: (3, 17, 256, 256) 即 (C, F, H, W)

    # 抽取 5 帧 (模拟 EchoVideoDataset 里的逻辑)
    total_frames = img_array.shape[1] # 帧维度 F 在 index=1
    # 均匀抽 5 帧的索引 (包含头尾，即 ED 和 ES)
    indices = np.linspace(0, total_frames - 1, 5, dtype=int)
    
    # 抽帧后形状: (3, 5, 256, 256)
    frames_5 = img_array[:, indices, :, :]
    
    # 根据你的要求：对轴做 swapaxes，使帧维在第一维 -> (5, 3, 256, 256) 即 (T, C, H, W)
    frames_5 = np.swapaxes(frames_5, 0, 1)
    
    # 转换为 Tensor，并增加 Batch 维度 -> (1, 5, 3, 256, 256) 即 (B, T, C, H, W)
    input_tensor = torch.from_numpy(frames_5).float().unsqueeze(0).to(device)
    print(f"  [数据处理] 喂入模型的张量维度: {input_tensor.shape}")

    # 伪造默认的 point prompt
    points = (torch.tensor([[[[128, 128]]]]).float().to(device), torch.tensor([[1]]).float().to(device))

    # ==================================================
    # 5. 执行显卡推理！
    # ==================================================
    print("⚡ 显卡轰鸣中，执行 5 帧时序序列分割...")
    with torch.no_grad():
        pred = model(input_tensor, points, None)
        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]
            
        # 沿着类别维度 (dim=1) 取 argmax，得到掩码
        # 此时 pred 维度通常是 (B, Classes, T, H, W) -> 取 argmax 后变成 (B, T, H, W)
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        # 最终 pred_mask 形状为 (5, 256, 256)

    # ==================================================
    # 6. 保存 5 帧掩码为新的 .npy 文件
    # ==================================================
    np.save(cli_args.output_mask, pred_mask)
    print(f"✅ 完美！5帧掩码已交付至: {cli_args.output_mask}")

if __name__ == "__main__":
    main()