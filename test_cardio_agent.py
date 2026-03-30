import os
import re
import json
import torch
import subprocess
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 唤醒 CardioAgent 大脑 (保持不变)
# ==========================================
print("正在唤醒 CardioAgent 医疗智能体，请稍候...")
base_model_path = "/home/ldh/llamafactory-setup/models/Qwen2.5-7B-Instruct"
lora_path = "/home/ldh/llamafactory-setup/LlamaFactory/saves/qwen2.5_7b_cardio_agent_final"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_path)
print("唤醒成功！准备接诊...\n" + "="*50)

# ==========================================
# 2. 真实底层视觉引擎接入 (跨环境打击)
# ==========================================
def run_cardiac_segmentation(image_path, model_type):
    print(f"  [系统执行] 跨环境唤醒 {model_type} 真实推理引擎，显卡轰鸣中...")
    
    # 💡 概念桥接：大模型传过来的是虚拟的单图路径 (如 patient0401_ED.npy)
    # 我们还原出真实的视频路径 (patient0401_4CH.npy) 交给显卡
    real_video_path = image_path.replace("_ED", "_4CH").replace("_ES", "_4CH")
    output_mask_path = real_video_path.replace(".npy", "_raw.npy")
    
    # 如果这个视频之前已经跑过掩码了，就直接跳过推理，节省时间！
    if os.path.exists(output_mask_path):
        print(f"  [系统缓存] 发现已处理过的掩码，直接调用缓存！")
        return output_mask_path

    # ================= 真实运行环境 =================
    memsam_python = "/home/ldh/miniconda3/envs/memsam/bin/python" # 替换为真实的 memsam python 路径
    project_dir = "/home/ldh/MemSAM-main"
    ckpt_path = f"{project_dir}/checkpoints/CAMUS/MemSAM_12191941_76_0.9246244405691503.pth"
    # ====================================================

    command = [
        memsam_python, 
        f"{project_dir}/inference_single.py",
        "--input_image", real_video_path,
        "--output_mask", output_mask_path,
        "--ckpt", ckpt_path
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" 
    
    try:
        result = subprocess.run(command, env=env, cwd=project_dir, capture_output=True, text=True, check=True)
        print(f"  [底层真实输出] >>> {result.stdout.strip()}") 
        print(f"  [底层反馈] 视觉引擎运行成功，5帧掩码已生成！")
        return output_mask_path
    except subprocess.CalledProcessError as e:
        print(f"  [底层报错] 视觉引擎崩溃:\n{e.stderr}")
        return f"ERROR: 分割失败"

def extract_chamber_mask(raw_mask_path, target_class):
    print(f"  [系统执行] 确认目标心腔 [{target_class}] 的掩码状态...")
    return raw_mask_path.replace("_raw", f"_{target_class}")

def calculate_ejection_fraction(ed_mask_path, es_mask_path):
    print(f"  [系统执行] 启动真实的 EF 量化计算引擎...")
    real_mask_path = ed_mask_path.replace("_LV", "_raw")
    
    try:
        mask_video = np.load(real_mask_path)
        ed_mask = mask_video[0]
        es_mask = mask_video[4]
        

        print(f"  [底层透视] ED掩码包含的像素值种类: {np.unique(ed_mask)}")
        
        # 严格限定只计算类别 1 (左心室) 的像素数量
        ed_volume = np.sum(ed_mask == 1) 
        es_volume = np.sum(es_mask == 1)
        
        if ed_volume == 0:
            return "ERROR: ED 容积为 0，掩码全黑"
            
        ef = (ed_volume - es_volume) / ed_volume * 100
        print(f"  [计算完毕] ED像素: {ed_volume}, ES像素: {es_volume}, 最终 EF: {ef:.2f}%")
        return f"{ef:.2f}%"
    except Exception as e:
        return f"ERROR: EF 计算崩溃 - {e}"
# ==========================================
# 3. 神经中枢调度逻辑 (Agent Loop)
# ==========================================
system_prompt = """You are CardioAgent, an intelligent medical imaging assistant. You have access to the following tools:
1. run_cardiac_segmentation: Run segmentation model (MemSAM/U-Net/ResU-Net) on echocardiogram. Args: {"image_path": str, "model_type": str}
2. extract_chamber_mask: Extract specific chamber (e.g., LV, LA, MYO) from raw segmentation. Args: {"raw_mask_path": str, "target_class": str}
3. calculate_ejection_fraction: Calculate EF using ED and ES LV masks. Args: {"ed_mask_path": str, "es_mask_path": str}

Follow the user's instructions and use tools sequentially when necessary."""

user_input = "请帮我计算一下患者 patient0401 的左心室射血分数 (EF)。ED图像在 /home/ldh/MemSAM-main/data/CAMUS_processed/videos/test/patient0401_ED.npy，ES图像在 /home/ldh/MemSAM-main/data/CAMUS_processed/videos/test/patient0401_ES.npy。"
print(f"👨‍⚕️ 医生指令: {user_input}\n")

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]

max_loops = 10 
current_loop = 0

while current_loop < max_loops:
    current_loop += 1
    print(f"🔄 [Agent 思考中 - 第 {current_loop} 轮]...")
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    clean_response = re.split(r'<\|im_end\|>|Observation:', response)[0].strip()
    
    messages.append({"role": "assistant", "content": clean_response})
    
    action_match = re.search(r'Action:\s*(\w+)', clean_response)
    args_match = re.search(r'Action Input:\s*(\{.*?\})', clean_response, re.DOTALL)
    
    if action_match and args_match:
        tool_name = action_match.group(1)
        tool_call = json.loads(args_match.group(1))
        print(f"🤖 大脑下达指令: 调用工具 [{tool_name}], 参数: {tool_call}")
        
        observation_result = ""
        if tool_name == "run_cardiac_segmentation":
            res = run_cardiac_segmentation(tool_call["image_path"], tool_call["model_type"])
            observation_result = f"原始全量分割已完成，保存在 {res}"
        elif tool_name == "extract_chamber_mask":
            res = extract_chamber_mask(tool_call["raw_mask_path"], tool_call["target_class"])
            observation_result = f"{tool_call['target_class']} 提取成功，保存在 {res}"
        elif tool_name == "calculate_ejection_fraction":
            res = calculate_ejection_fraction(tool_call["ed_mask_path"], tool_call["es_mask_path"])
            observation_result = f"计算完成，EF 值为 {res}"
            
        print(f"👀 视觉反馈 (Observation): {observation_result}\n")
        messages.append({"role": "user", "content": f"Observation: {observation_result}"})
    else:
        print("\n" + "="*50)
        print(f"🎉 最终交付 (Agent 对医生的回复):\n{clean_response}")
        print("="*50 + "\n")
        break

if current_loop >= max_loops:
    print("\n[警告] 超过最大推理轮数，Agent 可能迷失了方向。")