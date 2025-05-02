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
            self.whisper_model = whisper.load_model("turbo")
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
        
        # 检查文件是否存在
        if not os.path.exists(audio_path):
            error_msg = f"音频文件不存在: {audio_path}"
            logger.error(error_msg)
            print(f"错误: {error_msg}")
            return "【语音识别失败：未找到录音文件】"
        
        try:
            # 明确指定language="zh"，确保使用中文识别
            result = self.whisper_model.transcribe(
                audio_path, 
                language="zh",  # 指定语言为中文
                task="transcribe",  # 指定任务为转写
                initial_prompt="以下是简体中文的语音识别。"  # 添加引导提示，偏向简体输出
            )
            
            text = result["text"]
            
            # 繁体转简体
            if self.has_converter:
                text = self.converter.convert(text)
            
            logger.info(f"识别结果: {text}")
            return text
        except Exception as e:
            error_msg = f"语音识别出错: {str(e)}"
            logger.error(error_msg)
            return f"【语音识别失败：{str(e)}】"
    
    # 新增方法：使用大模型分析文本情感
    def analyze_text_with_llm(self, text):
        """使用大模型进行文本情感分析"""
        try:
            print("\n==== 大模型文本情感分析 ====")
            print(f"输入文本: {text}")
            
            # 构建提示词，让大模型进行情感分析
            prompt = f"""请分析以下文本的情感，只返回JSON格式的结果，包含以下情感类别的百分比(总和为100):
积极、消极、中性

同时分析出可能存在的具体情感，例如：高兴、愤怒、悲伤、厌恶、恐惧、惊讶等。
文本: "{text}"

JSON格式回答示例:
{{
  "情感分布": {{
    "积极": 30.5,
    "消极": 20.0,
    "中性": 49.5
  }},
  "主导情感": "中性",
  "具体情感": "平静"
}}

仅返回JSON格式，不要多余文字。"""
            
            # 使用大模型分析
            response, _ = self.qwen_model.chat(self.qwen_tokenizer, prompt, history=None)
            print(f"大模型情感分析原始返回: {response}")
            
            # 解析JSON结果
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                emotions_data = json.loads(json_str)
                
                # 提取情感分布
                emotions = emotions_data.get("情感分布", {})
                if not emotions:  # 兼容可能的不同JSON格式
                    emotions = {
                        "积极": emotions_data.get("积极", 0),
                        "消极": emotions_data.get("消极", 0),
                        "中性": emotions_data.get("中性", 0)
                    }
                
                # 确保有主要情感类别
                for key in ["积极", "消极", "中性"]:
                    if key not in emotions:
                        emotions[key] = 0
                        
                # 提取具体情感
                specific_emotion = emotions_data.get("具体情感", "未知")
                
                # 打印分析结果
                print("情感分析结果:")
                print(f"积极: {emotions['积极']:.2f}%")
                print(f"消极: {emotions['消极']:.2f}%")
                print(f"中性: {emotions['中性']:.2f}%")
                print(f"主导情感: {max(emotions, key=emotions.get)}")
                print(f"具体情感: {specific_emotion}")
                
                # 记录具体情感以便后续使用
                self.last_text_emotions = {
                    "distribution": emotions,
                    "specific": specific_emotion
                }
                
                return emotions
            else:
                print("无法解析大模型返回的情感分析结果，使用备用方法")
                # 使用备用方法
                return self._analyze_emotion_fallback(text)
                
        except Exception as e:
            print(f"大模型情感分析出错: {str(e)}")
            print("使用备用情感分析方法")
            # 使用备用方法
            return self._analyze_emotion_fallback(text)
    
    # 修改原来的方法为备用方法
    def _analyze_emotion_fallback(self, text):
        """基于规则的文本情感分析（作为备用）"""
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
        print("\n==== 备用文本情感分析 ====")
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
        
        return emotions
    
    # 修改主要的analyze_emotion方法，调用大模型分析
    def analyze_emotion(self, text):
        """分析文本情感，使用大模型"""
        logger.info(f"分析情感: {text}")
        
        # 使用大模型进行分析
        emotions = self.analyze_text_with_llm(text)
        
        logger.info(f"情感分析结果: {emotions}")
        return emotions

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
            
            # 保存原始情感分数以供后续使用
            self.last_audio_emotions = {}
            for i, label in enumerate(emotion_labels):
                if i < len(scores):
                    self.last_audio_emotions[label] = scores[i]
            
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
    
    # 新增多模态融合分析方法
    def analyze_multimodal_emotion(self, text, audio_path):
        """多模态情感分析：融合文本和音频的情感分析结果"""
        print("\n==== 多模态情感分析 ====")
        
        # 文本情感分析
        text_emotions = self.analyze_emotion(text)
        print("文本情感分析完成")
        
        # 音频情感分析
        audio_emotions = self.analyze_audio_emotion(audio_path)
        print("音频情感分析完成")
        
        # 权重设置 (可以根据实际情况调整)
        text_weight = 0.6   # 文本情感权重
        audio_weight = 0.4  # 音频情感权重
        
        # 融合情感分析结果
        combined_emotions = {}
        for emotion in ["积极", "消极", "中性"]:
            combined_emotions[emotion] = (
                text_emotions[emotion] * text_weight + 
                audio_emotions[emotion] * audio_weight
            )
        
        # 打印融合结果
        print("多模态情感融合结果:")
        print(f"积极: {combined_emotions['积极']:.2f}% (文本: {text_emotions['积极']:.2f}%, 音频: {audio_emotions['积极']:.2f}%)")
        print(f"消极: {combined_emotions['消极']:.2f}% (文本: {text_emotions['消极']:.2f}%, 音频: {audio_emotions['消极']:.2f}%)")
        print(f"中性: {combined_emotions['中性']:.2f}% (文本: {text_emotions['中性']:.2f}%, 音频: {audio_emotions['中性']:.2f}%)")
        print(f"主导情感: {max(combined_emotions, key=combined_emotions.get)}")
        print("====================\n")
        
        # 保存多模态融合结果，以便生成响应时使用
        self.last_multimodal_emotions = {
            "text": text_emotions,
            "audio": audio_emotions,
            "combined": combined_emotions,
            "text_specific": getattr(self, "last_text_emotions", {}).get("specific", "未知"),
            "audio_specific": self._get_specific_audio_emotion(audio_emotions)
        }
        
        return combined_emotions
    
    # 辅助方法：从音频情感获取具体情感
    def _get_specific_audio_emotion(self, emotions):
        """从音频情感分析结果中提取具体情感"""
        # 根据主导情感类型，选择最有可能的具体情感
        dominant = max(emotions, key=emotions.get)
        
        if dominant == "积极":
            return "高兴"
        elif dominant == "消极":
            # 如果有last_audio_emotions，可以更精确地判断
            if hasattr(self, "last_audio_emotions"):
                scores = self.last_audio_emotions
                if scores.get("生气(angry)", 0) > scores.get("悲伤(sad)", 0):
                    return "愤怒"
                else:
                    return "悲伤"
            return "不满"
        else:
            return "平静"
    
    def generate_response(self, text, emotions):
        """使用Qwen生成回复"""
        logger.info(f"为文本生成回复: {text}")
        
        # 获取主导情感
        dominant_emotion = max(emotions, key=emotions.get)
        
        # 构建情感文本描述
        emotion_text = "情感分析结果:\n"
        for emotion, value in emotions.items():
            emotion_text += f"{emotion}: {value:.2f}%\n"
        emotion_text += f"主导情感: {dominant_emotion}\n"
        
        # 检查是否有多模态融合结果
        specific_emotion = "未知"
        if hasattr(self, 'last_multimodal_emotions'):
            emotion_text += "\n多模态情感分析:\n"
            emotion_text += f"文本主要情感: {self.last_multimodal_emotions.get('text_specific', '未知')}\n"
            emotion_text += f"语音主要情感: {self.last_multimodal_emotions.get('audio_specific', '未知')}\n"
            specific_emotion = self.last_multimodal_emotions.get('text_specific', '未知')
        elif hasattr(self, 'last_text_emotions'):
            specific_emotion = self.last_text_emotions.get('specific', '未知')
            emotion_text += f"具体情感: {specific_emotion}\n"
        elif hasattr(self, 'last_audio_emotions'):
            audio_specific = self._get_specific_audio_emotion(emotions)
            emotion_text += f"语音情感: {audio_specific}\n"
            specific_emotion = audio_specific
        
        # 构建包含详细情感分析的提示词
        prompt = f"""客户说: "{text}"

客户情绪分析:
{emotion_text}

请根据客户的情绪状态生成一段专业、有同理心的客服回复，注意客户表现出的{specific_emotion}情感。
表现出对客户情绪的理解，并提供专业、积极的帮助。使用简体中文回复:"""
        
        # 生成回复
        response, _ = self.qwen_model.chat(self.qwen_tokenizer, prompt, history=None)
        
        # 确保回复是简体中文
        if hasattr(self, 'converter') and self.has_converter:
            response = self.converter.convert(response)
        
        logger.info(f"生成的回复: {response}")
        return response
