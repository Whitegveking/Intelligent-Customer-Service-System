# data
model = "lstm"

# dataset path 
data_path = "datasets/Emotion Speech Dataset"  # 更新为实际的数据集路径
class_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprise"]  # 精确匹配目录名称（首字母大写）
# 原始完整情绪标签集
# class_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]  
# class_labels:
#   - "01"  # Neutral
#   - "02"  # Calm
#   - "03"  # Happy
#   - "04"  # Sad
#   - "05"  # Angry
#   - "06"  # Fearful
#   - "07"  # Disgust
#   - "08"  # Surprised
nums_labels = 5  # 更新为实际的情绪类别数量

# feature path
feature_folder = "features/8-category" 
feature_method = "l"  #use librosa


# checkpoint path
checkpoint_path = "checkpoints/"  
checkpoint_name = "check_point_lstm"  

# train configs
epochs = 20  # number of epoch
batch_size = 32  # batch size
lr = 0.001  # learn rate

# model set
rnn_size = 128  # LSTM hidden layer size
hidden_size = 32
dropout = 0.5

params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,),
    'solver': 'adam',
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
