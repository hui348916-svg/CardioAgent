import os
import re
import json
import torch
import subprocess
import numpy as np
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 唤醒 CardioAgent 大脑 (保持全局加载)
# ==========================================
print("正在唤醒 CardioAgent 医疗智能体 (LangGraph版)...")
base_model_path = "/home/ldh/llamafactory-setup/models/Qwen2.5-7B-Instruct"
lora_path = "/home/ldh/llamafactory-setup/LlamaFactory/saves/qwen2.5_7b_cardio_agent_final"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(model, lora_path)
print("唤醒成功！底层逻辑图已编译完毕...\n" + "="*50)

# ==========================================
# 2. 底层工具箱 (CV + RAG) - 保持不变
# ==========================================
def run_cardiac_segmentation(image_path, model_type):
    real_video_path = image_path.replace("_ED", "_4CH").replace("_ES", "_4CH")
    output_mask_path = real_video_path.replace(".npy", "_raw.npy")
    if os.path.exists(output_mask_path): return output_mask_path
    memsam_python = "/home/ldh/miniconda3/envs/memsam/bin/python" 
    project_dir = "/home/ldh/MemSAM-main"
    ckpt_path = f"{project_dir}/checkpoints/CAMUS/MemSAM_12191941_76_0.9246244405691503.pth"
    command = [memsam_python, f"{project_dir}/inference_single.py", "--input_image", real_video_path, "--output_mask", output_mask_path, "--ckpt", ckpt_path]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0" 
    try:
        subprocess.run(command, env=env, cwd=project_dir, capture_output=True, text=True, check=True)
        return output_mask_path
    except Exception: return "ERROR: 分割失败"

def extract_chamber_mask(raw_mask_path, target_class):
    return raw_mask_path.replace("_raw", f"_{target_class}")

def calculate_ejection_fraction(ed_mask_path, es_mask_path):
    real_mask_path = ed_mask_path.replace("_LV", "_raw")
    try:
        mask_video = np.load(real_mask_path)
        ed_volume, es_volume = np.sum(mask_video[0] == 1), np.sum(mask_video[4] == 1)
        if ed_volume == 0: return "ERROR: ED 容积为 0，掩码全黑"
        ef = (ed_volume - es_volume) / ed_volume * 100
        return f"{ef:.2f}%"
    except Exception as e: return f"ERROR: EF 计算崩溃 - {e}"

def query_historical_medical_records(patient_id: str, query: str):
    try:
        client = chromadb.PersistentClient(path="./cardio_history_db")
        collection = client.get_collection(name="patient_records")
        results = collection.query(query_texts=[query], n_results=1, where={"patient_id": patient_id})
        if results['documents'] and results['documents'][0]: return results['documents'][0][0]
        return "未检索到该患者的相关历史病历。"
    except Exception as e: return f"检索失败: {e}"


# ==========================================
# 3. 🕸️ 终极形态：LangGraph 状态机定义
# ==========================================
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END

# [数据总线]：定义在节点之间流转的“状态字典”
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 历史对话，使用 add 使得每次返回的新消息会自动 append
    reflection_count: int                    # 熔断计数器
    final_answer: str                        # 最终交付报告

# [节点 A]：大脑推理引擎
def llm_node(state: AgentState):
    print("🔄 [Agent 推理节点] 思考中...")
    text = tokenizer.apply_chat_template(state["messages"], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    clean_response = re.split(r'<\|im_end\|>|Observation:', response)[0].strip()
    
    # 状态更新：只返回增量数据，LangGraph 会自动合入总状态
    return {"messages": [{"role": "assistant", "content": clean_response}]}

# [节点 B]：工具执行手臂
def tool_node(state: AgentState):
    last_message = state["messages"][-1]["content"]
    action_match = re.search(r'Action:\s*(\w+)', last_message)
    args_match = re.search(r'Action Input:\s*(\{.*?\})', last_message, re.DOTALL)
    
    observation_result = "解析工具参数失败"
    if action_match and args_match:
        tool_name = action_match.group(1)
        tool_call = json.loads(args_match.group(1))
        print(f"🤖 [工具执行节点] 触发调用: {tool_name}")
        
        if tool_name == "run_cardiac_segmentation":
            observation_result = f"原始全量分割已完成，保存在 {run_cardiac_segmentation(tool_call['image_path'], tool_call['model_type'])}"
        elif tool_name == "extract_chamber_mask":
            observation_result = f"{tool_call['target_class']} 提取成功，保存在 {extract_chamber_mask(tool_call['raw_mask_path'], tool_call['target_class'])}"
        elif tool_name == "calculate_ejection_fraction":
            observation_result = f"计算完成，EF 值为 {calculate_ejection_fraction(tool_call['ed_mask_path'], tool_call['es_mask_path'])}"
        elif tool_name == "query_historical_medical_records":
            observation_result = f"历史病历内容：\n{query_historical_medical_records(tool_call['patient_id'], tool_call['query'])}"
            
    print(f"👀 [视觉反馈] {observation_result}")
    return {"messages": [{"role": "user", "content": f"Observation: {observation_result}"}]}

# [节点 C]：主任医师反思与熔断器
def critic_node(state: AgentState):
    draft = state["messages"][-1]["content"]
    current_count = state.get("reflection_count", 0)
    print("\n🚦 [Critic 反思节点] 正在进行医疗安全校验...")
    
    critic_prompt = f"你是一位严苛的心内科主任。请审查下级医生的诊断结论：\n「{draft}」\n强制审查：1. 射血分数是否为负数？2. 射血分数是否缺失或极度异常？如果有错，回复 REJECT:[原因]。如果完全合理，仅回复 PASS。"
    
    inputs = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": critic_prompt}], tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
    critic_response = tokenizer.decode(model.generate(**inputs, max_new_tokens=100, temperature=0.01)[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    if "REJECT" in critic_response.upper():
        new_count = current_count + 1
        print(f"🚨 [审查未通过 - 第 {new_count} 次] {critic_response}")
        
        if new_count >= 2:
            print("🛑 [系统熔断] 反思次数超限，强制下线保护！")
            return {
                "reflection_count": 1, 
                "final_answer": "【系统自动中止】诊断失败。底层影像数据异常（疑似像素提取失败或EF逻辑谬误），为确保医疗安全，已拦截本次异常报告。请人工介入核查影像。"
            }
        else:
            feedback = f"【系统拦截】你的结论未通过审查！意见：{critic_response}。\n请立刻反思。如果底层数据已损坏，请直接向医生汇报失败，绝不要输出负数！"
            return {"messages": [{"role": "user", "content": feedback}], "reflection_count": 1}
    else:
        print("✅ [审查通过] 准许放行。")
        return {"final_answer": draft}

# [边 Edge]：路由判断逻辑
def route_after_llm(state: AgentState):
    last_message = state["messages"][-1]["content"]
    if "Action:" in last_message:
        return "continue_to_tool" # 如果有工具，去执行工具
    return "continue_to_critic"   # 如果没工具，去给主任医师审查

def route_after_critic(state: AgentState):
    if state.get("final_answer"): 
        return "end_process"      # 如果有了最终答案（通过审查或熔断），结束
    return "back_to_llm"          # 如果被打回重做，回到大脑

# ------------------------------------------
# 构建并编译有向无环图 (DAG)
# ------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("llm", llm_node)
workflow.add_node("tools", tool_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", route_after_llm, {"continue_to_tool": "tools", "continue_to_critic": "critic"})
workflow.add_edge("tools", "llm") # 工具执行完，必定回到大脑
workflow.add_conditional_edges("critic", route_after_critic, {"end_process": END, "back_to_llm": "llm"})

# 编译为可执行应用
app_graph = workflow.compile()


# ==========================================
# 4. 对外 API 入口
# ==========================================
def run_agent(patient_id: str, ed_path: str, es_path: str, instruction: str) -> str:
    system_prompt = """You are CardioAgent, an intelligent medical imaging assistant. You have access to tools:
1. run_cardiac_segmentation: {"image_path": str, "model_type": str}
2. extract_chamber_mask: {"raw_mask_path": str, "target_class": str}
3. calculate_ejection_fraction: {"ed_mask_path": str, "es_mask_path": str}
4. query_historical_medical_records: {"patient_id": str, "query": str}

【工作流要求】：算出 EF 后，必须调 query_historical_medical_records 查历史并进行横向对比分析，再输出结论。"""

    user_input = f"{instruction} 患者 {patient_id} 的射血分数。ED图像: {ed_path}，ES图像: {es_path}。"
    print(f"\n👨‍⚕️ [API 接收指令]: {user_input}")

    # 初始化状态词典
    initial_state = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "reflection_count": 0,
        "final_answer": ""
    }

    # 🚀 一键启动 LangGraph！recursion_limit 防止图死循环
    final_state = app_graph.invoke(initial_state, {"recursion_limit": 20})
    
    return final_state["final_answer"]
