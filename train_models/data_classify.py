import os
import shutil

# define emotion
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def organize_files(src_folder, dest_folder):
    # check and create destination folfer
    for emotion in emotion_map.values():
        emotion_folder = os.path.join(dest_folder, emotion)
        if not os.path.exists(emotion_folder):
            os.makedirs(emotion_folder)
            print(f"Create folder: {emotion_folder}")

    # all files from source folder
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.wav'):
                # name of third part is emotion tag
                parts = file.split('-')
                emotion_code = parts[2]  
                emotion_name = emotion_map.get(emotion_code, 'unknown')

                #dest name
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_folder, emotion_name, file)

                try:
                    # move
                    shutil.move(src_file, dest_file)
                    print(f"move file from: {src_file} to {dest_file}")
                except Exception as e:
                    print(f"move {src_file} fail: {e}")

# first classify first 
source_directory = "source" #set the source_directory
destination_directory = "datasets"
organize_files(source_directory, destination_directory)
