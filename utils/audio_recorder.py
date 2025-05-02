import pyaudio
import wave
import threading
import time
import os
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal


class AudioRecorder(QObject):
    recording_started = pyqtSignal()
    recording_finished = pyqtSignal(str)
    
    def __init__(self, channels=1, rate=16000, chunk=1024, format=pyaudio.paInt16):
        super().__init__()
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.format = format
        self.recording = False
        self.audio = pyaudio.PyAudio()
    
    def start_recording(self, output_dir="temp"):
        if self.recording:
            return
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用时间戳创建唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.output_file = os.path.join(output_dir, f"audio_{timestamp}.wav")
        
        self.recording = True
        
        # 在单独的线程中执行录音，避免阻塞主线程
        threading.Thread(target=self._record).start()
        self.recording_started.emit()
    
    def stop_recording(self):
        self.recording = False
    
    def _record(self):
        # 打开音频流
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        frames = []
        
        # 录制音频
        while self.recording:
            data = stream.read(self.chunk)
            frames.append(data)
        
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
        
        # 确保目录存在
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存录音文件
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        
        print(f"录音已保存到：{self.output_file}")
        
        # 发送完成信号
        self.recording_finished.emit(self.output_file)