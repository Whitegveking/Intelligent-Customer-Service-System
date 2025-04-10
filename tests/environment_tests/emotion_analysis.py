import torch
from transformers import pipeline

class ChineseEmotionAnalyzer:
    """简单的中文文本情感分析器"""
    
    def __init__(self):
        """初始化文本情感分析器"""
        print("初始化中文情感分析模型...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 使用轻量级的中文情感分析模型
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="uer/roberta-base-finetuned-jd-binary-chinese",
            device=0 if self.device == "cuda" else -1
        )
        print(f"情感分析模型加载完成，使用设备: {self.device}")
    
    def analyze(self, text):
        """分析文本情感
        
        Args:
            text (str): 需要分析的中文文本
            
        Returns:
            dict: 情感分析结果
        """
        result = self.sentiment_analyzer(text)[0]
        
        # 将结果转换为更友好的格式
        label = result["label"]
        sentiment = "正面" if label == "positive" else "负面"
        confidence = result["score"]
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_result": result
        }