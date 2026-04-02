# init_db.py
import chromadb

# 1. 在本地文件夹建一个数据库
client = chromadb.PersistentClient(path="./cardio_history_db")
collection = client.get_or_create_collection(name="patient_records")

# 2. 编造一条一年前的病历数据
patient_id = "patient0401"
historical_record = """
【超声心动图历史报告】
检查日期：2024年4月10日 (一年前)
患者ID：patient0401
主诉：乳腺癌术后，准备进行第二疗程阿霉素化疗。化疗前常规心功能评估。
超声所见：各心腔大小正常。左心室收缩功能良好。
量化指标：左心室舒张末期容积(EDV)正常，收缩末期容积(ESV)正常。经Simpson双平面法计算，射血分数 (EF) 为 68%。
诊断结论：心功能正常，未见明显节段性室壁运动异常，可耐受化疗。
"""

# 3. 把病历存进向量数据库
collection.add(
    documents=[historical_record],
    metadatas=[{"patient_id": patient_id, "year": "2024"}],
    ids=["record_0401_2024"]
)

print("✅ 历史病历已成功向量化并存入本地 ChromaDB 数据库！")