import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QTextEdit, QLabel, QProgressBar,
                            QSplitter, QFrame, QFileDialog, QMessageBox, 
                            QListWidget, QListWidgetItem, QScrollArea,
                            QGraphicsDropShadowEffect, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QBrush, QLinearGradient
from utils.audio_recorder import AudioRecorder


class BubbleWidget(QFrame):
    """美化版聊天气泡组件"""
    def __init__(self, text, is_customer=True, parent=None):
        super().__init__(parent)
        
        # 设置框架样式
        self.setFrameShape(QFrame.StyledPanel)
        self.setContentsMargins(0, 0, 0, 0)
        
        # 使用QSS设置气泡样式
        if is_customer:
            # 客户气泡 - 使用右对齐、更现代的蓝色
            self.setStyleSheet("""
                QFrame {
                    background-color: #2979FF;
                    border-radius: 15px;
                    border-top-right-radius: 5px;
                    color: white;
                    padding: 8px;
                    margin: 2px;
                }
                QLabel {
                    color: white;
                    background-color: transparent;
                }
            """)
        else:
            # 客服气泡 - 使用左对齐、淡灰色背景
            self.setStyleSheet("""
                QFrame {
                    background-color: #F5F5F5;
                    border-radius: 15px;
                    border-top-left-radius: 5px;
                    color: #333333;
                    padding: 8px;
                    margin: 2px;
                }
                QLabel {
                    color: #333333;
                    background-color: transparent;
                }
            """)
        
        # 创建阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # 布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # 发送者标签 - 使用更小、更精致的字体
        sender = QLabel("客户" if is_customer else "智能客服")
        sender.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(sender)
        
        # 消息文本 - 使用更易读的字体
        message = QLabel(text)
        message.setFont(QFont("Segoe UI", 10))
        message.setWordWrap(True)
        message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(message)
        
        # 设置最小和最大宽度
        self.setMinimumWidth(100)
        self.setMaximumWidth(500)


class MessageItem(QWidget):
    """消息项容器"""
    def __init__(self, text, is_customer=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 创建气泡小部件
        bubble = BubbleWidget(text, is_customer)
        
        # 根据是客户还是客服调整对齐方式
        if is_customer:
            layout.addStretch()
            layout.addWidget(bubble)
        else:
            layout.addWidget(bubble)
            layout.addStretch()


class StyledButton(QPushButton):
    """自定义美化按钮"""
    def __init__(self, text="", icon_name=None, parent=None):
        super(StyledButton, self).__init__(text, parent)
        
        # 设置基本样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #2979FF;
                color: white;
                border-radius: 18px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #448AFF;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        
        # 设置图标（如果提供）
        if icon_name:
            # 这里假设icons目录存在，你需要创建这个目录并放入相应图标
            icon_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")
            icon_path = os.path.join(icon_dir, f"{icon_name}.png")
            if os.path.exists(icon_path):
                self.setIcon(QIcon(icon_path))
                self.setIconSize(QSize(20, 20))
        
        # 设置固定高度
        self.setFixedHeight(36)


class WorkerThread(QThread):
    """后台处理线程，避免UI卡顿"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, model_manager, text="", audio_path=None):
        super().__init__()
        self.model_manager = model_manager
        self.text = text
        self.audio_path = audio_path
        
    def run(self):
        try:
            self.progress.emit(10)
        
            if self.audio_path:
                # 处理语音
                self.progress.emit(20)
                # 音频转文字
                text = self.model_manager.recognize_speech(self.audio_path)
                self.text = text
                self.progress.emit(30)
            
                # 使用emotion2vec分析音频情感
                emotions = self.model_manager.analyze_audio_emotion(self.audio_path)
                self.progress.emit(50)
            else:
                # 使用基于规则的方法分析文本情感
                self.progress.emit(30)
                emotions = self.model_manager.analyze_emotion(self.text)
                self.progress.emit(50)
        
            # 生成回复
            response = self.model_manager.generate_response(self.text, emotions)
            self.progress.emit(90)
            
            # 返回结果
            result = {
                "text": self.text,
                "emotions": emotions,
                "response": response
            }
            self.progress.emit(100)
            self.finished.emit(result)
        
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.audio_recorder = AudioRecorder()
        self.current_audio_path = None
        self.worker = None
        self.chat_history = []
        
        # 设置窗口
        self.setWindowTitle("智能客服系统")
        self.setMinimumSize(900, 650)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            QScrollArea {
                border: none;
                background-color: white;
            }
            QLabel {
                color: #333333;
            }
            QProgressBar {
                border: none;
                border-radius: 10px;
                background-color: #E0E0E0;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2979FF;
                border-radius: 10px;
            }
            QTextEdit {
                border: 1px solid #E0E0E0;
                border-radius: 15px;
                padding: 10px;
                background-color: white;
                selection-background-color: #BBDEFB;
            }
        """)
        
        # 创建UI组件
        self.init_ui()
        
        # 连接信号和槽
        self.connect_signals()
        
        # 添加欢迎消息
        self.add_message("您好！我是您的智能客服助手，很高兴为您服务。请问有什么可以帮助您的？", is_customer=False)
        
    def init_ui(self):
        # 创建中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 顶部标题栏
        title_bar = QFrame()
        title_bar.setFixedHeight(60)
        title_bar.setStyleSheet("""
            QFrame {
                background-color: #2979FF;
                border-bottom: 1px solid #1565C0;
            }
        """)
        
        title_layout = QHBoxLayout(title_bar)
        
        # 标题标签
        title_label = QLabel("智能客服系统")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_layout.addWidget(title_label)
        
        # 状态标签，靠右对齐
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: white;")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_layout.addWidget(self.status_label)
        
        main_layout.addWidget(title_bar)
        
        # 主内容区域
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(15)
        
        # 聊天区域 (使用滚动区域包装)
        chat_container = QWidget()
        chat_container.setStyleSheet("background-color: white;")
        self.chat_layout = QVBoxLayout(chat_container)
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(15)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(chat_container)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border-radius: 15px;
                background-color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #F0F0F0;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #BDBDBD;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9E9E9E;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        scroll_area.setGraphicsEffect(shadow)
        
        content_layout.addWidget(scroll_area, stretch=1)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        content_layout.addWidget(self.progress_bar)
        
        # 底部控制区域
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("background-color: white; border-radius: 15px;")
        bottom_layout = QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(15, 15, 15, 15)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        bottom_frame.setGraphicsEffect(shadow)
        
        # 文本输入框
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请输入您的问题...")
        self.input_text.setFixedHeight(80)
        self.input_text.setFont(QFont("Segoe UI", 11))
        bottom_layout.addWidget(self.input_text)
        
        # 按钮区域
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)
        
        # 录音按钮
        self.btn_record = StyledButton("语音", "mic")
        self.btn_record.setToolTip("开始录音")
        buttons_layout.addWidget(self.btn_record)
        
        # 停止录音按钮
        self.btn_stop = StyledButton("停止", "stop")
        self.btn_stop.setToolTip("停止录音")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border-radius: 18px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        buttons_layout.addWidget(self.btn_stop)
        
        # 发送按钮
        self.btn_send = StyledButton("发送", "send")
        self.btn_send.setToolTip("发送消息")
        buttons_layout.addWidget(self.btn_send)
        
        # 调整按钮布局
        buttons_layout.addStretch()
        bottom_layout.addLayout(buttons_layout)
        
        content_layout.addWidget(bottom_frame)
        
        main_layout.addWidget(content_widget)

    def connect_signals(self):
        # 录音按钮
        self.btn_record.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        
        # 发送按钮
        self.btn_send.clicked.connect(self.send_message)
        
        # 输入框回车键
        self.input_text.installEventFilter(self)
        
        # 音频录制信号
        self.audio_recorder.recording_started.connect(self.on_recording_started)
        self.audio_recorder.recording_finished.connect(self.on_recording_finished)
    
    def eventFilter(self, obj, event):
        if obj == self.input_text and event.type() == event.KeyPress:
            if event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)
    
    def start_recording(self):
        self.status_label.setText("正在录音...")
        # 创建temp目录
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        # 让AudioRecorder自动创建唯一的文件名
        self.audio_recorder.start_recording(temp_dir)
    
    def stop_recording(self):
        self.audio_recorder.stop_recording()
        self.status_label.setText("录音已停止，等待处理...")
    
    def on_recording_started(self):
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_send.setEnabled(False)
        
        # 添加录音中的动画效果
        self.recording_timer = QTimer(self)
        self.recording_timer.timeout.connect(self.update_recording_status)
        self.recording_timer.start(500)  # 每500ms更新一次
        self.recording_dots = 0
    
    def update_recording_status(self):
        # 动态显示录音状态
        self.recording_dots = (self.recording_dots + 1) % 4
        dots = "." * self.recording_dots
        self.status_label.setText(f"正在录音{dots}")
    
    def on_recording_finished(self, path):
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_send.setEnabled(True)
        self.current_audio_path = path
        
        # 停止录音动画
        if hasattr(self, 'recording_timer'):
            self.recording_timer.stop()
        
        # 录音完成后再处理
        if path and os.path.exists(path):
            self.status_label.setText("录音已完成，正在处理...")
            self.process_input(audio_only=True)
        else:
            self.status_label.setText("录音失败")
            QMessageBox.warning(self, "录音错误", "录音保存失败，请重试！")
    
    def send_message(self):
        text = self.input_text.toPlainText().strip()
        if text:
            # 添加客户消息到聊天区域
            self.add_message(text, is_customer=True)
            # 清空输入框
            self.input_text.clear()
            # 处理消息
            self.process_input(text=text)
        
    def process_input(self, text=None, audio_only=False):
        # 禁用按钮防止重复点击
        self.btn_send.setEnabled(False)
        self.btn_record.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 获取输入文本
        if text is None:
            text = self.input_text.toPlainText().strip()
            self.input_text.clear()
            
        # 如果是纯音频处理，不需要文本
        if audio_only:
            text = ""
        
        # 检查是否有输入
        if not text and not self.current_audio_path:
            QMessageBox.warning(self, "输入错误", "请输入文本或进行语音录制！")
            self.btn_send.setEnabled(True)
            self.btn_record.setEnabled(True)
            return
        
        # 如果是语音输入且没有显示文本，显示"正在处理语音..."
        if self.current_audio_path and not text:
            self.add_message("(正在处理语音...)", is_customer=True)
        
        # 创建并启动工作线程
        self.worker = WorkerThread(
            self.model_manager, 
            text=text, 
            audio_path=self.current_audio_path if not text else None
        )
        
        # 连接信号
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_results)
        self.worker.error.connect(self.handle_error)
        
        # 启动处理
        self.status_label.setText("处理中...")
        self.worker.start()
    
    def add_message(self, text, is_customer=True, emotions=None):
        # 创建消息项
        message_item = MessageItem(text, is_customer)
        
        # 添加到布局
        self.chat_layout.addWidget(message_item)
        
        # 如果有情感分析结果，记录下来但不显示
        if emotions:
            self.chat_history.append({
                "text": text,
                "is_customer": is_customer,
                "emotions": emotions
            })
        else:
            self.chat_history.append({
                "text": text,
                "is_customer": is_customer
            })
            
        # 滚动到底部
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def scroll_to_bottom(self):
        # 获取滚动区域
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            # 滚动到底部
            vsb = scroll_area.verticalScrollBar()
            vsb.setValue(vsb.maximum())
    
    @pyqtSlot(int)
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    @pyqtSlot(dict)
    def handle_results(self, results):
        # 如果是语音输入，更新之前的"处理中"消息
        if self.current_audio_path and not results["text"] in [item["text"] for item in self.chat_history if item.get("is_customer", False)]:
            # 移除最后一条客户消息（如果是处理中的消息）
            last_item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if last_item and isinstance(last_item.widget(), MessageItem):
                last_widget = last_item.widget()
                self.chat_layout.removeWidget(last_widget)
                last_widget.deleteLater()
                if self.chat_history and self.chat_history[-1]["is_customer"]:
                    self.chat_history.pop()
            
            # 添加识别出的文本作为客户消息
            self.add_message(results["text"], is_customer=True)
        
        # 添加客服回复
        self.add_message(results["response"], is_customer=False, emotions=results["emotions"])
        
        # 重置状态
        self.status_label.setText("就绪")
        self.progress_bar.setValue(0)
        self.btn_send.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.current_audio_path = None  # 清除当前音频路径
    
    @pyqtSlot(str)
    def handle_error(self, error_msg):
        QMessageBox.critical(self, "处理错误", f"发生错误: {error_msg}")
        self.status_label.setText("处理出错")
        self.btn_send.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.current_audio_path = None