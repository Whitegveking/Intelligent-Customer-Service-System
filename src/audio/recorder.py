import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import logging

# 导入项目配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import AUDIO_CONFIG

logger = logging.getLogger("audio.recorder")

class AudioRecorder:
    """音频录制工具类"""
    
    def __init__(self, sample_rate=None, channels=None):
        """
        初始化录音器
        
        Args:
            sample_rate (int, optional): 采样率
            channels (int, optional): 声道数
        """
        # 使用配置文件中的设置或默认值
        self.sample_rate = sample_rate or AUDIO_CONFIG.get("sample_rate", 16000)
        self.channels = channels or AUDIO_CONFIG.get("channels", 1)
        self.dtype = 'float32'
        
        # 创建临时目录
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def record(self, duration):
        """
        录制指定时长的音频
        
        Args:
            duration (int): 录音时长(秒)
            
        Returns:
            numpy.ndarray: 录制的音频数据
        """
        try:
            logger.info(f"开始录制 {duration} 秒音频...")
            
            # 录音
            audio_data = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=self.channels, 
                              dtype=self.dtype)
            sd.wait()  # 等待录音完成
            
            logger.info("录音完成")
            return audio_data
            
        except Exception as e:
            logger.error(f"录音过程出错: {e}")
            raise
    
    def save_audio(self, audio_data, filename=None):
        """
        保存音频数据到文件
        
        Args:
            audio_data (numpy.ndarray): 音频数据
            filename (str, optional): 文件名，如不指定则生成临时文件名
            
        Returns:
            str: 保存的文件路径
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.temp_dir, f"recording_{timestamp}.wav")
            
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"音频已保存至: {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"保存音频文件时出错: {e}")
            raise
    
    def record_and_save(self, duration, filename=None):
        """
        录制并保存音频
        
        Args:
            duration (int): 录音时长(秒)
            filename (str, optional): 保存的文件名
            
        Returns:
            tuple: (音频数据, 文件路径)
        """
        audio_data = self.record(duration)
        file_path = self.save_audio(audio_data, filename)
        return audio_data, file_path