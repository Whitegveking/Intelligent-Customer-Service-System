import os
import sys
import time
import whisper
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入情感分析模块
from emotion_analysis import ChineseEmotionAnalyzer

# FFmpeg 配置
ffmpeg_path = "E:\\ffmpeg-7.0.2-essentials_build\\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
    print(f"已添加FFmpeg路径: {ffmpeg_path}")
else:
    print(f"警告: 指定的FFmpeg路径不存在: {ffmpeg_path}")
    print("请修改ffmpeg_path变量为正确的FFmpeg安装路径")

def test_emotion():
    """测试语音情感分析功能"""
    print("语音情感分析测试\n")
    
    # 1. 加载模型
    print("正在加载Whisper模型...")
    model_size = "turbo"  # 使用medium可以加快速度
    whisper_model = whisper.load_model(model_size)
    print(f"Whisper {model_size} 模型加载完成")
    print(f"设备: {whisper_model.device}")
    
    # 2. 初始化情感分析模型
    emotion_analyzer = ChineseEmotionAnalyzer()
    
    # 3. 提供测试选项
    print("\n请选择测试方式：")
    print("1. 从音频文件中分析情感")
    print("2. 录制新的音频并分析情感")
    choice = input("请输入选择 (1/2): ")
    
    if choice == '1':
        analyze_from_file(whisper_model, emotion_analyzer)
    elif choice == '2':
        analyze_from_recording(whisper_model, emotion_analyzer)
    else:
        print("无效选择，退出测试")

def analyze_from_file(whisper_model, emotion_analyzer):
    """从文件分析情感"""
    audio_path = input("请输入音频文件路径 (例如: test.wav): ")
    
    # 确保文件存在
    if not os.path.isabs(audio_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_path = os.path.join(script_dir, audio_path)
    
    if not os.path.exists(audio_path):
        print(f"文件不存在: {audio_path}")
        return
    
    print(f"\n开始分析音频文件: {audio_path}")
    
    # 1. 语音转文字
    print("正在进行语音识别...")
    result = whisper_model.transcribe(audio_path)
    transcript = result["text"]
    language = result["language"]
    
    print("\n语音识别结果:")
    print("-" * 50)
    print(transcript)
    print(f"检测到的语言: {language}")
    
    # 2. 文本情感分析
    print("\n正在分析情感...")
    emotion_result = emotion_analyzer.analyze(transcript)
    
    print("\n情感分析结果:")
    print("-" * 50)
    print(f"文本内容: '{transcript}'")
    print(f"情感倾向: {emotion_result['sentiment']}")
    print(f"置信度: {emotion_result['confidence']:.2f}")

def analyze_from_recording(whisper_model, emotion_analyzer):
    """录制并分析情感"""
    # 录音参数
    sample_rate = 16000
    seconds = int(input("请输入录音时长(秒): ") or "5")
    
    print(f"\n将录制{seconds}秒音频，请在提示后开始讲话...")
    input("准备好后按Enter键开始录音...")
    
    print("正在录音...(请说话)")
    
    # 录音
    audio_data = sd.rec(int(seconds * sample_rate), 
                        samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # 等待录音完成
    
    print("录音完成！")
    
    # 保存临时文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_file = os.path.join(script_dir, f"recording_{timestamp}.wav")
    sf.write(temp_file, audio_data, sample_rate)
    
    print(f"\n开始分析录制的音频...")
    
    # 1. 语音转文字
    print("正在进行语音识别...")
    result = whisper_model.transcribe(temp_file)
    transcript = result["text"]
    language = result["language"]
    
    print("\n语音识别结果:")
    print("-" * 50)
    print(transcript)
    print(f"检测到的语言: {language}")
    
    # 2. 文本情感分析
    print("\n正在分析情感...")
    emotion_result = emotion_analyzer.analyze(transcript)
    
    print("\n情感分析结果:")
    print("-" * 50)
    print(f"文本内容: '{transcript}'")
    print(f"情感倾向: {emotion_result['sentiment']}")
    print(f"置信度: {emotion_result['confidence']:.2f}")
    
    # 询问是否保留录音文件
    keep = input("\n是否保留录音文件？(y/n): ")
    if keep.lower() != 'y':
        os.remove(temp_file)
        print(f"已删除临时录音文件 {temp_file}")
    else:
        print(f"录音文件已保存为 {temp_file}")

if __name__ == "__main__":
    test_emotion()