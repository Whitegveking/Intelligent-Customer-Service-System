import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper
import logging
# 导入FunASR
from funasr import AutoModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_manager")

class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self._init_models()
    
    def _init_models(self):
        try:
            # 1. 加载Whisper模型
            logger.info("加载Whisper模型...")
            self.whisper_model = whisper.load_model("small")
            logger.info("Whisper模型加载完成")
            
            # 2. 加载emotion2vec模型（使用FunASR）
            logger.info("加载emotion2vec情感分析模型...")
            self.emotion_model = AutoModel(
                model="iic/emotion2vec_plus_base",  # 使用官方支持的模型ID
                hub="hf"  # 国内用户可以使用"ms"或"modelscope"，海外用户使用"hf"或"huggingface"
            )
            logger.info("情感分析模型加载完成")
            
            # 3. 加载Qwen模型
            logger.info("加载Qwen模型...")
            model_name = "Qwen/Qwen-1_8B-Chat"
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.qwen_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Qwen模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def recognize_speech(self, audio_path):
        """使用Whisper识别语音"""
        logger.info(f"识别音频: {audio_path}")
        result = self.whisper_model.transcribe(audio_path, language="zh")
        text = result["text"]
        logger.info(f"识别结果: {text}")
        return text
    
    # 修改analyze_emotion方法，添加更详细的打印
    def analyze_emotion(self, text):
        """分析文本情感或音频情感"""
        logger.info(f"分析情感: {text}")
        
        # 情感词典
        positive_words = ["满意", "喜欢", "感谢", "好", "优秀", "快", "棒", "赞", "方便", "效率", 
                        "不错", "可以", "很好", "帮助", "满足", "高兴"]
        negative_words = ["不满", "差", "慢", "退款", "投诉", "不行", "垃圾", "骗", "失望", "麻烦",
                        "问题", "错误", "难用", "糟糕", "生气", "恼火"]
        
        # 计算积极和消极词的出现次数
        positive_matches = [word for word in positive_words if word in text]
        negative_matches = [word for word in negative_words if word in text]
        
        positive_count = len(positive_matches)
        negative_count = len(negative_matches)
        
        # 打印匹配到的情感词
        print("\n==== 文本情感分析 ====")
        print(f"输入文本: {text}")
        print(f"识别到的积极词: {positive_matches if positive_matches else '无'}")
        print(f"识别到的消极词: {negative_matches if negative_matches else '无'}")
        
        # 计算情感得分
        if positive_count == 0 and negative_count == 0:
            # 没有明显情感词，默认中性占主导
            emotions = {
                "积极": 20.0,
                "消极": 20.0,
                "中性": 60.0
            }
            print("未检测到明显情感，使用默认中性情感")
        else:
            total = positive_count + negative_count
            if total == 0:
                total = 1  # 避免除以零
            
            positive_score = (positive_count / total) * 80  # 最高80分
            negative_score = (negative_count / total) * 80  # 最高80分
            neutral_score = 100 - positive_score - negative_score
            
            emotions = {
                "积极": positive_score,
                "消极": negative_score,
                "中性": neutral_score
            }
        
        # 打印情感分析结果
        print("情感分析结果:")
        print(f"积极: {emotions['积极']:.2f}%")
        print(f"消极: {emotions['消极']:.2f}%")
        print(f"中性: {emotions['中性']:.2f}%")
        print(f"主导情感: {max(emotions, key=emotions.get)}")
        print("====================\n")
        
        logger.info(f"情感分析结果: {emotions}")
        return emotions

    # 修复analyze_audio_emotion方法的缩进问题并添加打印输出
    def analyze_audio_emotion(self, audio_path):
        """使用emotion2vec分析音频情感"""
        try:
            print("\n==== 音频情感分析 ====")
            print(f"分析音频: {audio_path}")
            
            # 使用FunASR的emotion2vec模型提取情感
            rec_result = self.emotion_model.generate(
                audio_path, 
                output_dir="./outputs", 
                granularity="utterance", 
                extract_embedding=False
            )
            
            # 调试信息
            logger.info(f"rec_result type: {type(rec_result)}")
            logger.info(f"rec_result content: {rec_result}")
            print(f"情感识别原始结果: {rec_result}")
            
            # 适应不同的返回格式
            if isinstance(rec_result, dict) and 'scores' in rec_result:
                # 原来预期的格式
                scores = rec_result['scores']
                print("结果格式: 字典包含scores")
            elif isinstance(rec_result, list):
                # 新的返回格式是列表
                if len(rec_result) > 0 and isinstance(rec_result[0], dict) and 'scores' in rec_result[0]:
                    # 如果是列表中包含字典
                    scores = rec_result[0]['scores']
                    print("结果格式: 列表包含字典")
                else:
                    # 假设列表本身就是分数
                    scores = rec_result
                    print("结果格式: 列表直接包含分数")
            else:
                # 无法识别的格式，使用默认值
                scores = [0.1, 0.1, 0.1, 0.2, 0.3, 0.0, 0.1, 0.1, 0.0]
                print("结果格式: 未知格式，使用默认值")
            
            # 确保scores是列表且长度足够
            if not isinstance(scores, list) or len(scores) < 7:
                scores = [0.1, 0.1, 0.1, 0.2, 0.3, 0.0, 0.1, 0.1, 0.0]
                print("分数格式错误或长度不足，使用默认值")
            
            print(f"处理后的情感分数: {scores}")
            
            # 情感标签映射
            emotion_labels = [
                "生气(angry)", "厌恶(disgusted)", "恐惧(fearful)", 
                "高兴(happy)", "中性(neutral)", "其他(other)", 
                "悲伤(sad)", "惊讶(surprised)", "未知(unknown)"
            ]
            
            # 打印各情感分数
            print("各情感原始分数:")
            for i, label in enumerate(emotion_labels):
                if i < len(scores):
                    print(f"{label}: {scores[i]:.4f}")
            
            # 转换成客服系统使用的三类情感
            emotions = {
                "积极": scores[3] * 100,  # happy
                "消极": (scores[0] + scores[1] + scores[2] + scores[6]) * 100,  # angry + disgusted + fearful + sad
                "中性": scores[4] * 100  # neutral
            }
            
            # 标准化情感得分，确保总和为100
            total = sum(emotions.values())
            if total > 0:
                for key in emotions:
                    emotions[key] = (emotions[key] / total) * 100
            
            # 打印最终情感分析结果
            print("情感分析结果:")
            print(f"积极: {emotions['积极']:.2f}%")
            print(f"消极: {emotions['消极']:.2f}%")
            print(f"中性: {emotions['中性']:.2f}%")
            print(f"主导情感: {max(emotions, key=emotions.get)}")
            print("====================\n")
            
            return emotions
        except Exception as e:
            logger.error(f"音频情感分析出错: {str(e)}")
            # 记录详细错误信息和调用栈
            import traceback
            logger.error(traceback.format_exc())
            # 返回默认情感
            print("音频情感分析出错:")
            print(str(e))
            print("使用默认情感值")
            default_emotions = {"积极": 20.0, "消极": 20.0, "中性": 60.0}
            print("====================\n")
            return default_emotions
    
    def generate_response(self, text, emotions):
        """使用Qwen生成回复"""
        logger.info(f"为文本生成回复: {text}")
        
        # 获取主导情感
        dominant_emotion = max(emotions, key=emotions.get)
        
        # 具体情感映射
        detailed_emotion_map = {
            "积极": ["满意", "开心", "高兴", "兴奋", "愉快"],
            "消极": ["愤怒", "不满", "失望", "焦虑", "悲伤", "烦恼"],
            "中性": ["平静", "冷静", "中立"]
        }
        
        # 检查是否有原始情感得分数据（来自analyze_audio_emotion方法）
        detailed_emotions = ""
        raw_emotions_available = False
        
        # 获取最近一次的音频分析结果
        try:
            if hasattr(self, 'last_audio_emotions') and isinstance(self.last_audio_emotions, dict):
                detailed_emotions = "具体情感强度:\n"
                emotion_types = [
                    ("生气(angry)", "愤怒"),
                    ("厌恶(disgusted)", "厌恶"),
                    ("恐惧(fearful)", "恐惧"),
                    ("高兴(happy)", "高兴"),
                    ("悲伤(sad)", "悲伤"),
                    ("惊讶(surprised)", "惊讶")
                ]
                
                # 提取原始分数
                scores = []
                for label, chinese_name in emotion_types:
                    for key, value in self.last_audio_emotions.items():
                        if key.startswith(label.split("(")[0]):  # 匹配标签前缀
                            scores.append((chinese_name, value))
                            break
                
                # 按分数降序排序
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # 添加前三种主要情感
                for i, (emotion_name, score) in enumerate(scores[:3]):
                    if score > 0.05:  # 只显示显著的情感
                        detailed_emotions += f"{emotion_name}: {score*100:.2f}%\n"
                        raw_emotions_available = True
        except Exception as e:
            logger.error(f"获取详细情感失败: {str(e)}")
        
        # 如果没有详细情感数据，使用基于主导情感的猜测
        if not raw_emotions_available:
            import random
            detailed_emotion = random.choice(detailed_emotion_map[dominant_emotion])
            detailed_emotions = f"推测的具体情感: {detailed_emotion}"
        
        # 构建情感文本描述
        emotion_text = "情感分析结果:\n"
        for emotion, value in emotions.items():
            emotion_text += f"{emotion}: {value:.2f}%\n"
        emotion_text += f"主导情感: {dominant_emotion}\n"
        emotion_text += detailed_emotions
        
        # 构建包含详细情感分析的提示词
        prompt = f"""客户说: "{text}"

    客户情绪分析:
    {emotion_text}

    请根据客户的情绪状态生成一段专业、有同理心的客服回复，注意客户可能表现出的具体情感。
    表现出对客户情绪的理解，并提供专业、积极的帮助:"""
        
        # 生成回复
        response, _ = self.qwen_model.chat(self.qwen_tokenizer, prompt, history=None)
        
        # 确保回复是简体中文
        if hasattr(self, 'converter') and self.has_converter:
            response = self.converter.convert(response)
        
        logger.info(f"生成的回复: {response}")
        return response