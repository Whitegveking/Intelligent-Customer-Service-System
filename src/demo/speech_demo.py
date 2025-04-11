"""语音转文本功能演示脚本"""

import os
import sys
import time

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio import WhisperSTT, AudioRecorder

def main():
    """主函数"""
    print("==== 智能客服系统 - 语音转文本演示 ====\n")
    
    # 初始化语音转文本引擎
    print("初始化语音转文本引擎...")
    stt = WhisperSTT()
    stt.load_model()
    
    # 初始化录音器
    recorder = AudioRecorder()
    
    # 菜单循环
    while True:
        print("\n请选择操作:")
        print("1. 从音频文件转录")
        print("2. 录制并转录音频")
        print("3. 退出")
        choice = input("请输入选择 (1-3): ")
        
        if choice == '1':
            audio_path = input("\n请输入音频文件路径: ")
            language = input("指定语言(可选, 如'zh'表示中文, 直接回车自动检测): ")
            
            language = language if language else None
            
            print("\n开始转录...")
            result = stt.transcribe_file(audio_path, language)
            
            if "error" in result and result["error"]:
                print(f"转录失败: {result['error']}")
            else:
                print("\n转录结果:")
                print("-" * 50)
                print(result["text"])
                print("-" * 50)
                print(f"检测到的语言: {result['language']}")
                print(f"处理时间: {result.get('process_time', 0):.2f}秒")
        
        elif choice == '2':
            try:
                duration = int(input("\n请输入录音时长(秒): ") or "5")
                language = input("指定语言(可选, 如'zh'表示中文, 直接回车自动检测): ")
                
                language = language if language else None
                
                print(f"\n准备录音 {duration} 秒, 请准备...")
                time.sleep(1)
                print("开始录音...(请说话)")
                
                # 录音
                audio_data = recorder.record(duration)
                print("录音完成！")
                
                # 转录
                print("\n开始转录...")
                result = stt.transcribe_audio_data(audio_data, language=language)
                
                if "error" in result and result["error"]:
                    print(f"转录失败: {result['error']}")
                else:
                    print("\n转录结果:")
                    print("-" * 50)
                    print(result["text"])
                    print("-" * 50)
                    print(f"检测到的语言: {result['language']}")
                    print(f"处理时间: {result.get('process_time', 0):.2f}秒")
            
            except ValueError:
                print("无效的录音时长，请输入整数")
        
        elif choice == '3':
            print("\n谢谢使用，再见！")
            break
        
        else:
            print("\n无效选择，请重试")

if __name__ == "__main__":
    main()