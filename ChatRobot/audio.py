import os
from cv2 import log
import librosa
import soundfile as sf
import pyaudio
import wave
import webrtcvad
import logging
import numpy as np
import struct


# webrtcvad 人声检测效果太差，vad 选择自制算法用能量计算

def Bytes2Int16Slice(feature):
    x = []
    for i in range(len(feature)//2):
        data = feature[i * 2: (i * 2) + 2]
        a = struct.unpack('h', data)
        x.append(a[0])
    return x

class Audio:
    def __init__(self) -> None:
        self.p = pyaudio.PyAudio()
        
        self.vad = webrtcvad.Vad(1)
        self.sample_rate=16000
        
        
        self.chunk = 2048
        self.mean_db = 0
        self.mean_db_path = os.path.relpath("mean_db.npy")
        
        self.FORMAT = pyaudio.paInt16
        
        self.init_log()
    
    def init_log(self):
        self.logger = logging.getLogger("Audio")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        # 日志输出格式
        self.formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)
    
    
    def init(self):
        if os.path.exists(self.mean_db_path):
            self.logger.info(f"初始化文件存在： {self.mean_db_path}， 从磁盘中加载")
            self.mean_db = np.load(self.mean_db_path)
            return
        
        self.logger.info("接下来开始录制5s的环境静音，用于初始化录音系统")
        self.logger.info("输入任意键回车后开始, Ctrl + C取消")
        input()
        stream = self.p.open(format=self.FORMAT,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        count = 0
        count_max = int( 5 * self.sample_rate / self.chunk)
        means = []
        self.logger.info("开始采集环境录音")
        while count < count_max:
            frame = stream.read(self.chunk)      # 读出声卡缓冲区的音频数据
            mean = self.frame_mean(frame)
            means.append(mean)
            count += 1
        
        self.mean_db = max(means)
        self.logger.info(f"响度平均值 max: {max(means)}, min: {min(means)}")
        self.logger.info(f"初始化平均响度 {max(means)}")
        
        # 序列化保存
        np.save(self.mean_db_path, self.mean_db)
        self.logger.info(f"序列化保存路径: {os.path.realpath(self.mean_db_path)}")
    
    def frame_mean(self, frame):
        x = bytes(frame)
        x = Bytes2Int16Slice(x)
        x = np.array(x, dtype='float32')
        x = np.abs(x)
        return np.mean(x)

    def is_speech(self, frame):
        mean = self.frame_mean(frame)
        if mean > self.mean_db:
            return True
        else:
            return False
    
    def record(self, outpath):        
        stream = self.p.open(format=self.FORMAT,
                             channels=1,
                             rate=self.sample_rate,
                             input=True,
                             frames_per_buffer=self.chunk)
        # 新建一个列表，用来存储采样到的数据
        record_buf = []
        count = 0
        vad_flag = False
        
        max_cnt = 200
        max_silence_cnt = int(1 * self.sample_rate / self.chunk)
        silence_cnt = 0
        is_useful = False
        is_first = True
        
        self.logger.info("开始录音")
        
        while not vad_flag:
            audio_data = stream.read(self.chunk)      # 读出声卡缓冲区的音频数据
            active = self.is_speech(audio_data)
            
            if active:
                # 数据有效
                silence_cnt = 0
                is_useful = True
                if is_first:
                    self.logger.info("检测到声音，开始录制")
                    is_first = False
                
            else:
                silence_cnt += 1
                # self.logger.info("检测到静音")
                if silence_cnt > max_silence_cnt and is_useful:
                    # 持续静音大于2s
                    vad_flag = True
                    self.logger.info("结束录音")
                    break

            if is_useful:
                record_buf.append(audio_data)
                
            count += 1
            if count > max_cnt:
                break
            
        stream.stop_stream()
        stream.close()
        
        if not is_useful:
            return None
        else:
            wf = wave.open(outpath, 'wb')          # 创建一个音频文件，名字为“01.wav"
            wf.setnchannels(1)                      # 设置声道数为2
            wf.setsampwidth(2)                      # 设置采样深度为
            wf.setframerate(16000)                  # 设置采样率为16000
            # 将数据写入创建的音频文件
            wf.writeframes("".encode().join(record_buf))
            # 写完后将文件关闭
            wf.close()
            return outpath
    
    def play(self, audio_file):
        f = wave.open(audio_file,"rb")
        
        #open stream
        stream = self.p.open(format = self.p.get_format_from_width(f.getsampwidth()),
                        channels = f.getnchannels(),
                        rate = f.getframerate(),
                        output = True)
        #read data
        data = f.readframes(self.chunk)
        
        #paly stream
        while len(data) > 0:
            stream.write(data)
            data = f.readframes(self.chunk)
        
        #stop stream
        stream.stop_stream()
        stream.close()

if __name__ == '__main__':
    audio = Audio()
    audio.init()
    audio.record("../wav/record.wav")
    