from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from test_cardio_agent import run_agent

app = FastAPI(
    title="CardioAgent Core API",
    description="基于 Qwen2.5 与 MemSAM 的医疗影像端到端量化分析微服务",
    version="1.0.0"
)

# 定义前端发送过来的 JSON 数据格式（设置了默认值，方便你在网页里直接点测试）
class AnalyzeRequest(BaseModel):
    patient_id: str = "patient0401"
    ed_image_path: str = "/home/ldh/MemSAM-main/data/CAMUS_processed/videos/test/patient0401_ED.npy"
    es_image_path: str = "/home/ldh/MemSAM-main/data/CAMUS_processed/videos/test/patient0401_ES.npy"
    instruction: str = "请帮我计算一下"

# 暴露一个 POST 接口供外部调用
@app.post("/api/v1/analyze_ef")
async def analyze_ejection_fraction(request: AnalyzeRequest):
    print(f"\n🚀 [API 路由] 接收到新任务：计算患者 {request.patient_id} 的 EF 值...")
    
    try:
        # 这里会触发大模型的 while 循环和显卡的 subprocess 推理
        agent_result = run_agent(
            patient_id=request.patient_id,
            ed_path=request.ed_image_path,
            es_path=request.es_image_path,
            instruction=request.instruction
        )
        
        # 将 Agent 生成的最终文本组装成标准的 JSON 返回给前端
        return {
            "code": 200,
            "status": "success",
            "patient_id": request.patient_id,
            "data": {
                "ef_report": agent_result
            }
        }
    except Exception as e:
        print(f"❌ [API 崩溃] 运行出错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)