import os
import whisper
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# 添加FFmpeg路径到环境变量
# 替换为您系统中FFmpeg的实际安装路径
ffmpeg_path = "E:\\ffmpeg-7.0.2-essentials_build\\bin"  # 通常安装位置，请根据您的实际安装路径修改
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
    print(f"已添加FFmpeg路径: {ffmpeg_path}")
else:
    print(f"警告: 指定的FFmpeg路径不存在: {ffmpeg_path}")
    print("请修改ffmpeg_path变量为正确的FFmpeg安装路径")

# 检查FFmpeg是否可用
import subprocess
try:
    subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    print("FFmpeg检测成功!")
except Exception as e:
    print(f"FFmpeg检测失败: {e}")
    print("提示: 请确保FFmpeg已安装并添加到PATH中")
    print("您可以通过设置正确的ffmpeg_path变量来解决此问题")
    # 不退出，让用户看到错误信息

def test_whisper():
    """测试原始OpenAI Whisper模型功能"""
    print("OpenAI Whisper语音识别模型测试\n")
    
    # 1. 加载模型
    model_size = "large"  # 可选：tiny, base, small, medium, large, large-v2, turbo
    print(f"正在加载Whisper {model_size} 模型...")
    start_time = time.time()
    model = whisper.load_model(model_size)
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时：{load_time:.2f}秒")
    print(f"设备: {model.device}")
    
    # 提供两种测试选项
    print("\n请选择测试方式：")
    print("1. 从音频文件中识别语音")
    print("2. 录制新的音频并识别")
    choice = input("请输入选择 (1/2): ")
    
    if choice == '1':
        test_from_file(model)
    elif choice == '2':
        test_from_recording(model)
    else:
        print("无效选择，退出测试")

def test_from_file(model):
    """使用原始Whisper从文件测试"""
    audio_path = input("请输入音频文件路径 (例如: test.wav): ")
    
    if not os.path.exists(audio_path):
        print(f"文件不存在: {audio_path}")
        return
    
    print(f"\n开始转录音频文件: {audio_path}")
    print("处理中...")
    
    # 删除单独的语言检测调用，直接进行转录
    # 运行完整转录
    start_time = time.time()
    result = model.transcribe(audio_path)
    process_time = time.time() - start_time
    
    print(f"\n转录完成! 处理时间: {process_time:.2f}秒")
    print("\n识别结果:")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)
    print(f"检测到的语言: {result['language']}")

def test_from_recording(model):
    """录制并使用原始Whisper测试"""
    # 录音参数
    sample_rate = 16000
    seconds = 5
    
    print(f"\n将录制{seconds}秒音频，请在提示后开始讲话...")
    input("准备好后按Enter键开始录音...")
    
    print("正在录音...(请说话)")
    
    # 录音
    audio_data = sd.rec(int(seconds * sample_rate), 
                        samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 等待录音完成
    
    print("录音完成！")
    
    # 保存为临时文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = f"recording_{timestamp}.wav"
    sf.write(temp_file, audio_data, sample_rate)
    
    print("正在识别...")
    
    # 删除单独的语言检测调用
    # 运行转录
    start_time = time.time()
    result = model.transcribe(temp_file)
    process_time = time.time() - start_time
    
    print(f"\n转录完成! 处理时间: {process_time:.2f}秒")
    print("\n识别结果:")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)
    print(f"检测到的语言: {result['language']}")
    
    # 询问是否保留录音文件
    keep = input("\n是否保留录音文件？(y/n): ")
    if keep.lower() != 'y':
        os.remove(temp_file)
        print(f"已删除临时录音文件 {temp_file}")
    else:
        print(f"录音文件已保存为 {temp_file}")

def demo_language_detection(model, audio):
    """演示低级别的语言检测API"""
    print("\n正在检测语言...")
    
    # 处理音频
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    1
    # 检测语言
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    
    print(f"检测到的语言: {detected_lang} (置信度: {probs[detected_lang]:.2f})")
    
    return detected_lang

def demo_lower_level_api(model, audio_path):
    """演示更低级别的API使用"""
    print("\n演示Whisper低级API使用:")
    
    # 加载并处理音频
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    # 检测语言
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    print(f"检测到的语言: {detected_lang}")
    
    # 解码
    options = whisper.DecodingOptions(language=detected_lang, fp16=False)
    result = whisper.decode(model, mel, options)
    
    print("通过低级API识别的文本:")
    print(result.text)

if __name__ == "__main__":
    test_whisper()