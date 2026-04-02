# 🩺 CardioAgent: An LLM-Driven Autonomous Agent for Echocardiogram Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)
![FastAPI](https://img.shields.io/badge/Microservice-FastAPI-009688)
![Medical AI](https://img.shields.io/badge/Domain-Medical_AI-red)
![License](https://img.shields.io/badge/License-MIT-green)

CardioAgent 是一个专为医学超声影像（Echocardiogram）量化分析设计的企业级多智能体系统。
针对传统原生多模态大模型（如 GPT-4V）在医学像素级计算中存在严重“数学幻觉”且缺乏透明度的痛点，本项目通过 **LangGraph 状态机** 将 LLM 逻辑中枢与专有 CV 视觉引擎深度解耦，实现了从“医学影像物理分割”到“纵向时序病历预警”的全自动、零幻觉安全闭环。

## ✨ 核心工程亮点 (Highlights)

### 1. 🧠 基于 LangGraph 的工业级状态机 (DAG Workflow)
摒弃传统脆弱的 `while` 循环嵌套，采用有向无环图（DAG）重构 Agent。将“LLM 推理”、“CV 工具调度”与“Critic 审查”深度解耦为独立节点（Node），实现了高可维护性、状态持久化与防上下文污染的复杂智能体工作流。

### 2. ⚡ Subprocess IPC 显存物理隔离机制 (OOM Prevention)
针对医疗边缘设备单卡显存受限痛点，**首创基于子进程通信的“按需挂载”调度机制**。大模型（Qwen2.5）常驻显存时，重型 CV 模型（MemSAM）被系统级动态拉起，计算完毕瞬间释放 100% 显卡资源，成功在消费级显卡上打通 LLM 与重型 CV 的高并发调度。

### 3. 🛡️ Actor-Critic 医疗安全熔断器 (Circuit Breaker)
为确保医疗场景 `0 致命幻觉`，在系统交付最终报告前置入独立的 Critic 审查节点。
* **常识拦截：** 对生成的射血分数（EF）等指标进行生理学常识交叉验证（如拦截负数 EF）。
* **安全降级：** 遭遇底层影像损坏时，自动触发反思重试机制；超限（2次）后触发系统级熔断，强制拦截异常报告并请求人工介入。

### 4. 📚 基于本地轻量级 RAG 的时序纵向推理
集成 ChromaDB 构建本地患者历史病历向量库。Agent 在算出当前临床指标后，通过 Function Calling 自主检索既往文本病历，进行横向时序对比（如自动预警化疗前后的心功能断崖式衰退），赋予智能体长时记忆综合诊断能力。

### 5. 🌐 企业级微服务化交付 (FastAPI)
底层复杂的异步图调度逻辑被 FastAPI 完全封装，对外提供高可用 RESTful API 与 Swagger 可视化交互界面，具备无缝接入医院 HIS 前端系统的能力。
## 💻 运行演示 (Execution Demo)

当输入患者超声影像并触发计算时，Agent 底层基于 LangGraph 的思考链路如下：

🔄 [Agent 推理节点] 思考中...

🤖 [工具执行节点] 触发调用: run_cardiac_segmentation (跨进程唤醒视觉引擎...)

👀 [视觉反馈] 原始全量分割已完成

🔄 [Agent 推理节点] 思考中...

🤖 [工具执行节点] 触发调用: query_historical_medical_records (RAG 检索...)

👀 [视觉反馈] 发现一年前病史：化疗前 EF 为 68%

🚦 [Critic 反思节点] 正在进行医疗安全校验...

🚨 [审查未通过 - 第 1 次] REJECT: 提取像素过少，EF 呈现异常值，存在分割失败风险。

🛑 [系统熔断] 反思次数超限，强制下线保护！生成医疗异常拦截报告！

## 🚀 极速启动 (Quick Start)

### 1. 环境依赖配置
建议使用 Conda 创建独立虚拟环境：
```bash
conda create -n cardio_agent python=3.10
conda activate cardio_agent
pip install -r requirements.txt
```
### 2. 初始化 RAG 时序病历库
```bash
# 生成本地 ChromaDB 向量数据卷
cd rag
python init_db.py
```
### 3. 启动微服务内核
```bash
cd agent
python api_server.py
```
服务启动后，默认监听 8000 端口。请在浏览器访问 http://127.0.0.1:8000/docs 进入 Swagger UI 进行可视化 API 测试。

==================================================
