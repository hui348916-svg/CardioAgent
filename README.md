# 🩺 CardioAgent: An LLM-Driven Autonomous Agent for Echocardiogram Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c) ![LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-SFT-green)

本项目是一个面向医疗影像领域的端到端智能体（Agent）调度系统。系统以 Qwen2.5-7B 为语言中枢大脑，通过深度定义工具调用逻辑（Function Calling），自主调度底层的 MemSAM / U-Net 等视觉大模型，实现心脏多腔室的零代码交互式分割与高级临床量化指标（如射血分数 EF）的自动计算。

## ✨ 核心工程亮点 (Highlights)

- **🧠 医疗 Agent 专属微调 (SFT)**：依托 LLaMA-Factory 框架，融合开源 Glaive 通用指令与数百条垂直心血管医学 ShareGPT 语料，完成两阶段 LoRA 微调。使模型具备 100% 格式对齐的 JSON 工具调用能力。
- **🚀 物理级解耦架构 (防 OOM)**：首创 **Subprocess 跨虚拟环境调度机制**。LLM 调度器与底层重度 CV 引擎运行在隔离的 Conda 环境中，按需拉起、用完释放，彻底根除联合部署时的 CUDA 显存溢出与包依赖冲突。
- **🔗 端到端闭环能力**：打通了从“自然语言意图理解” -> “多步工具调用规划” -> “CV 模型显卡推理” -> “.npy 视频流容积计算”的完整链路。

## 💻 运行演示 (Execution Demo)

当医生输入自然语言指令时，CardioAgent 会自主规划并跨环境调用显卡完成多步计算：

```text
==================================================
👨‍⚕️ 医生指令: 请帮我计算一下患者 patient0401 的左心室射血分数 (EF)。ED图像在 /data/videos/test/patient0401_ED.npy，ES图像在 /data/videos/test/patient0401_ES.npy。

🔄 [Agent 思考中 - 第 1 轮]...
🤖 大脑下达指令: 调用工具 [run_cardiac_segmentation], 参数: {'image_path': '/data/videos/test/patient0401_ED.npy', 'model_type': 'MemSAM'}
  [系统执行] 跨环境唤醒 MemSAM 真实推理引擎，显卡轰鸣中...
  [底层反馈] 视觉引擎运行成功，5帧掩码已生成！
👀 视觉反馈 (Observation): 原始全量分割已完成，保存在 /data/videos/test/patient0401_4CH_raw.npy

🔄 [Agent 思考中 - 第 2 轮]...
🤖 大脑下达指令: 调用工具 [extract_chamber_mask], 参数: {'raw_mask_path': '/data/videos/test/patient0401_4CH_raw.npy', 'target_class': 'LV'}
  [系统执行] 确认目标心腔 [LV] 的掩码状态...
👀 视觉反馈 (Observation): LV 提取成功，保存在 /data/videos/test/patient0401_4CH_LV.npy

... (经过连续 5 轮自主逻辑调度与显卡计算) ...

🎉 最终交付 (Agent 对医生的回复):
计算完毕。我分别对该患者的舒张末期(ED)和收缩末期(ES)影像进行了 MemSAM 分割并提取了左心室(LV)区域。经计算，该患者的射血分数 (EF) 为 **50.00%**。
==================================================