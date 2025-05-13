
from deepseek import KnowledgeBase
import os

kb = KnowledgeBase(api_key="your_key")
kb.create(
    name="⼼⾎管疾病库",
    documents=["heart_disease.pdf", "treatment_guide.docx"],
    description="三甲医院内部诊疗标准",
    access_level="private"
)
host = os.getenv('OLLAMA_HOST') or 'http://ps.co6co.top:65098'
