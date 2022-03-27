import paddlehub as hub
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.asr.infer import ASRExecutor
import os
import librosa
import soundfile as sf
from audio import Audio
import logging
    
    

class Robot:
    def __init__(self) -> None:
        self.asr_model = ASRExecutor()
        self.chat_model = hub.Module(name="plato-mini")
        self.tts_model = TTSExecutor()
        
        self.audio = Audio()
        self.asr_name = "conformer_wenetspeech"
        self.am_name = "fastspeech2_csmsc"
        self.voc_name = "mb_melgan_csmsc"
        
        self.logger = logging.getLogger("Robot")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        # 日志输出格式
        self.formatter = logging.Formatter('[%(asctime)s] - %(levelname)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        
    
    def init(self): 
        # tts init
        self.logger.info("语音合成 服务初始化")
        text = "今天天气真好"
        wavDir = os.path.join(os.path.dirname(__file__),'../wav')
        self.wavDir = wavDir
        if not os.path.exists(wavDir):
            os.makedirs(wavDir)
        outpath = os.path.join(wavDir, 'demo.wav')
        self.tts_model(text, am=self.am_name, voc=self.voc_name, output=outpath)
        
        # asr init
        self.logger.info("语音识别 服务初始化")
        wav,_ = librosa.load(outpath, sr=16000)
        outpath_16k = os.path.join(wavDir, 'demo_16k.wav')
        sf.write(outpath_16k, wav, 16000)
        result = self.asr_model(outpath_16k, model=self.asr_name,lang='zh',
                 sample_rate=16000)
        self.logger.info(f"初始化识别结果: {result}")
        
        # audio init
        self.logger.info("录音功能初始化")
        self.audio.init()
        
        self.tts_outpath = os.path.join(self.wavDir, 'tts.wav')
        self.record_path = os.path.join(self.wavDir, 'record.wav')
        
        self.logger.info("初始化服务完成")
        
    def start(self):
        self.guide()
        # asr 识别
        while True:
            audio_file = self.audio.record(self.record_path)
            if audio_file:
                asr_res = self.speech2text(audio_file)
                self.logger.info(f"asr： {asr_res}")
                
                if len(asr_res) > 0:
                    # 过滤空音频
                    self.leave(asr_res)
                    # 识别结果给chat
                    chat_result = self.chat(asr_res)
                    self.logger.info(f"chat 回应:  {chat_result}")
                    
                    # chat 结果合成
                    self.text2speech(chat_result, self.tts_outpath)
                    
                    # pyaudio 播放音频
                    self.audio.play(self.tts_outpath)
                    audio_file = None
        

    def speech2text(self, audio_file):
        self.asr_model.preprocess(self.asr_name, audio_file)
        self.asr_model.infer(self.asr_name)
        res = self.asr_model.postprocess()
        return res
    
    def text2speech(self, text, outpath):
        self.tts_model.infer(text=text, lang="zh", am=self.am_name, spk_id=0)
        res = self.tts_model.postprocess(output=outpath)
        return res

    def chat(self, text):
        result = self.chat_model.predict([text])
        return result[0]
    
    def guide(self):
        text = "欢迎使用智能语音机器人，敲击回车后进入对话，跟我说不聊了就可以结束对话！"
        self.logger.info(text)
        self.text2speech(text, self.tts_outpath)
        self.audio.play(self.tts_outpath)
        input("敲击回车进入对话")
        self.logger.info("智能语音机器人启动")
    
    def leave(self, text):
        if "不聊了" in text:
            text = "感谢您的使用，期待与您下次相遇！"
            self.logger.info(text)
            self.text2speech(text, self.tts_outpath)
            self.audio.play(self.tts_outpath)
            exit(1)
        else:
            return False


if __name__ == '__main__':
    robot = Robot()
    robot.init()
    robot.start()
    