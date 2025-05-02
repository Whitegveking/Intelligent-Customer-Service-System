import sys
import os

# 设置ffmpeg路径
ffmpeg_path = r"C:\Program Files (x86)\ffmpeg-7.0.2-essentials_build\bin"
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from models.model_manager import ModelManager


if __name__ == "__main__":
    # 确保模型保存目录存在
    os.makedirs("local_models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)  # FunASR输出目录
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 设置风格
    
    # 加载模型管理器
    model_manager = ModelManager()
    
    # 创建并显示主窗口
    window = MainWindow(model_manager)
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())