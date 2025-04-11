import os
import time
import whisper
import logging
from datetime import datetime
import soundfile as sf

# 添加简繁转换库
try:
    from opencc import OpenCC
    cc = OpenCC('t2s')  # 繁体转简体
    HAS_OPENCC = True
except ImportError:
    print("未安装OpenCC库，将使用备用方案。建议安装：pip install opencc-python-reimplemented")
    try:
        import zhconv
        HAS_ZHCONV = True
    except ImportError:
        print("未安装zhconv库，将无法将繁体中文转换为简体中文。建议安装：pip install zhconv")
        HAS_ZHCONV = False
    HAS_OPENCC = False

# 导入项目配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from config import WHISPER_CONFIG, FFMPEG_PATH
except ImportError:
    # 默认配置
    WHISPER_CONFIG = {"model_size": "medium", "device": None}
    FFMPEG_PATH = "E:\\ffmpeg-7.0.2-essentials_build\\bin"
    print("未找到配置文件，使用默认配置")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("speech_to_text")

class WhisperSTT:
    """基于OpenAI Whisper的语音转文本引擎"""
    
    def __init__(self, model_size=None, device=None, to_simplified=True):
        """
        初始化Whisper语音转文本引擎
        
        Args:
            model_size (str, optional): 模型大小，可选值：tiny, base, small, medium, large, large-v2, turbo
            device (str, optional): 运行设备，None表示自动选择
            to_simplified (bool): 是否将繁体中文转换为简体中文，默认为True
        """
        # 使用配置文件中的设置或传入的参数
        self.model_size = model_size or WHISPER_CONFIG.get("model_size", "medium")
        self.device = device or WHISPER_CONFIG.get("device")
        self.model = None
        self.is_loaded = False
        self.to_simplified = to_simplified
        
        # 确保FFmpeg已配置
        self._setup_ffmpeg()
    
    def _setup_ffmpeg(self):
        """设置FFmpeg环境变量"""
        if os.path.exists(FFMPEG_PATH):
            os.environ["PATH"] = FFMPEG_PATH + os.pathsep + os.environ["PATH"]
            logger.info(f"已添加FFmpeg路径: {FFMPEG_PATH}")
        else:
            logger.warning(f"FFmpeg路径不存在: {FFMPEG_PATH}")
            logger.warning("请确保安装了FFmpeg并在config.py中设置正确的路径")
            
        # 验证FFmpeg可用性
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("FFmpeg检测成功!")
        except Exception as e:
            logger.error(f"FFmpeg检测失败: {e}")
            logger.error("Whisper模型需要FFmpeg支持，请确保正确安装")
    
    def _convert_traditional_to_simplified(self, text):
        """将繁体中文转换为简体中文"""
        if not self.to_simplified:
            return text
            
        if HAS_OPENCC:
            return cc.convert(text)
        elif HAS_ZHCONV:
            return zhconv.convert(text, 'zh-cn')
        else:
            logger.warning("未安装繁简转换库，无法进行转换，返回原始文本")
            return text
    
    def load_model(self):
        """加载Whisper模型"""
        if not self.is_loaded:
            logger.info(f"正在加载Whisper {self.model_size} 模型...")
            start_time = time.time()
            
            try:
                self.model = whisper.load_model(self.model_size, device=self.device)
                self.is_loaded = True
                load_time = time.time() - start_time
                logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
                logger.info(f"使用设备: {self.model.device}")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        
        return self.model
    
    def transcribe_file(self, audio_path, language=None):
        """
        从音频文件转录文本
        
        Args:
            audio_path (str): 音频文件路径
            language (str, optional): 指定语言代码，如'zh'表示中文，None表示自动检测
            
        Returns:
            dict: 包含转录结果和元数据的字典
        """
        if not self.is_loaded:
            self.load_model()
        
        if not os.path.exists(audio_path):
            error_msg = f"音频文件不存在: {audio_path}"
            logger.error(error_msg)
            return {"error": error_msg, "text": "", "language": None}
        
        logger.info(f"开始转录文件: {audio_path}")
        start_time = time.time()
        
        try:
            # 转录选项
            options = {
                "initial_prompt": "以下是普通话的说话内容" if language == "zh" else None
            }
            if language:
                options["language"] = language
                logger.info(f"使用指定语言: {language}")
                
            result = self.model.transcribe(audio_path, **options)
            process_time = time.time() - start_time
            
            # 将繁体中文转换为简体中文
            if result["language"] == "zh" and self.to_simplified:
                original_text = result["text"]
                result["text"] = self._convert_traditional_to_simplified(original_text)
                logger.info("已将繁体中文转换为简体中文")
                result["original_text"] = original_text  # 保留原始文本
            
            logger.info(f"转录完成，耗时: {process_time:.2f}秒")
            logger.info(f"检测到的语言: {result['language']}")
            
            # 添加元数据
            result["process_time"] = process_time
            result["file_path"] = audio_path
            
            return result
            
        except Exception as e:
            logger.error(f"转录过程出错: {e}")
            return {"error": str(e), "text": "", "language": None}
    
    def transcribe_audio_data(self, audio_data, sample_rate=16000, language=None):
        """
        转录内存中的音频数据
        
        Args:
            audio_data (numpy.ndarray): 音频数据数组
            sample_rate (int): 采样率
            language (str, optional): 指定语言代码
            
        Returns:
            dict: 包含转录结果和元数据的字典
        """
        if not self.is_loaded:
            self.load_model()
        
        # 创建临时文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "temp")
        
        # 确保临时目录存在
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f"recording_{timestamp}.wav")
        
        try:
            # 保存临时文件
            sf.write(temp_file, audio_data, sample_rate)
            logger.info(f"已保存临时音频文件: {temp_file}")
            
            # 调用文件转录
            result = self.transcribe_file(temp_file, language)
            
            # 删除临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"已删除临时文件: {temp_file}")
                
            return result
            
        except Exception as e:
            logger.error(f"转录内存音频数据出错: {e}")
            
            # 确保删除临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return {"error": str(e), "text": "", "language": None}