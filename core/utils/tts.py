import asyncio
import logging
import os
import json
import uuid
import base64
from datetime import datetime
import edge_tts
import numpy as np

import requests
import torch
# 动态将项目根目录添加到系统路径，确保顶层绝对导入能正常寻址
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.utils.util import read_config, get_project_dir
from pydub import AudioSegment
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTS(ABC):
    def __init__(self, config, delete_audio_file):
        self.delete_audio_file = delete_audio_file
        self.output_file = config.get("output_file")
        self.delete_audio_file = delete_audio_file

    @abstractmethod
    def generate_filename(self):
        pass

    # def to_tts(self, text):
    #     tmp_file = self.generate_filename()
    #     try:
    #         max_repeat_time = 5
    #         while not os.path.exists(tmp_file) and max_repeat_time > 0:
    #             asyncio.run(self.text_to_speak(text, tmp_file))
    #             if not os.path.exists(tmp_file):
    #                 max_repeat_time = max_repeat_time - 1
    #                 logger.error(f"语音生成失败: {text}:{tmp_file}，再试{max_repeat_time}次")

    #         return tmp_file
    #     except Exception as e:
    #         logger.info(f"Failed to generate TTS file: {e}")
    #         return None

    async def to_tts(self, text):
        """生成语音文件（异步版本）"""
        tmp_file = self.generate_filename()
        try:
            max_repeat_time = 5
            while not os.path.exists(tmp_file) and max_repeat_time > 0:
                # 直接 await 异步方法
                await self.text_to_speak(text, tmp_file)
                
                if not os.path.exists(tmp_file):
                    max_repeat_time = max_repeat_time - 1
                    logger.error(f"语音生成失败: {text}:{tmp_file}，再试{max_repeat_time}次")
            if not os.path.exists(tmp_file):
                logger.error(f"语音生成最终失败: {text}")
                return None
            return tmp_file
        except Exception as e:
            logger.error(f"Failed to generate TTS file: {e}")
            return None

    @abstractmethod
    async def text_to_speak(self, text, output_file):
        pass

    def get_audio_duration(self, audio_file_path):
        """获取音频文件时长（秒）"""
        if audio_file_path is None:
            logger.error("audio_file_path is None")
            return 0
        
        # 获取文件后缀名
        file_type = os.path.splitext(audio_file_path)[1]
        if file_type:
            file_type = file_type.lstrip('.')

        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_file_path)
            duration = waveform.shape[1] / sample_rate
        except Exception as e:
            logger.warning(f"使用torchaudio加载音频失败: {e}, 尝试使用pydub")
            try:
                audio = AudioSegment.from_file(audio_file_path, format=file_type)
                duration = len(audio) / 1000.0
            except Exception as e:
                logger.error(f"加载音频文件失败: {e}")
                return 0

        return duration


class EdgeTTS(TTS):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.voice = config.get("voice")
        self.proxy = config.get("proxy")

    def generate_filename(self, extension=".mp3"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    async def text_to_speak(self, text, output_file):
        max_retries = 3
        # 检查是否配置了语音
        if not self.voice:
            logger.error("未配置语音，无法进行语音合成")
            raise Exception("未配置语音，请检查配置文件中的TTS.voice设置")
            
        logger.info(f"使用语音 {self.voice} 进行语音合成")
        
        for attempt in range(max_retries):
                try:
                    logger.info(f"尝试使用语音 {self.voice} 进行转换 (尝试 {attempt+1}/{max_retries})")
                    # 添加随机延迟，避免请求过于频繁
                    if attempt > 0:
                        delay = (attempt + 1) * 2  # 2秒, 4秒, 6秒
                        logger.info(f"添加 {delay} 秒延迟以避免请求过于频繁")
                        await asyncio.sleep(delay)
                    
                    # 创建Communicate对象时添加代理支持
                    if self.proxy:
                        logger.info(f"使用代理: {self.proxy}")
                        communicate = edge_tts.Communicate(text, voice=self.voice, proxy=self.proxy)
                    else:
                        communicate = edge_tts.Communicate(text, voice=self.voice)
                    await communicate.save(output_file)
                    logger.info(f"成功使用语音 {self.voice} 转换文本")
                    return
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"使用语音 {self.voice} 转换失败: {error_str}")
                    
                    # 如果是403错误，可能是认证问题
                    if "403" in error_str:
                        logger.error("403错误，可能是认证问题，请检查API密钥或代理设置")
                    elif "Connection" in error_str or "timeout" in error_str:
                        # 连接错误或超时，等待后重试
                        logger.warning("连接错误或超时，等待后重试")
                        await asyncio.sleep(1 * (attempt + 1))
                    elif "rate limit" in error_str.lower() or "too many requests" in error_str.lower():
                        # 速率限制错误，增加等待时间
                        wait_time = 10 * (attempt + 1)
                        logger.warning(f"达到速率限制，等待 {wait_time} 秒后重试")
                        await asyncio.sleep(wait_time)
                    else:
                        # 其他错误，可能是网络问题，等待后重试
                        logger.warning("其他错误，可能是网络问题，等待后重试")
                        await asyncio.sleep(1 * (attempt + 1))
        
        # 所有尝试都失败，记录详细的错误信息
        logger.error(f"使用语音 {self.voice} 进行语音合成失败，已尝试 {max_retries} 次")
        # 尝试生成一个简单的测试音频作为备选方案
        try:
            logger.info("尝试生成测试音频作为备选方案")
            test_text = "语音合成测试"
            communicate = edge_tts.Communicate(test_text, voice=self.voice, proxy=self.proxy)
            await communicate.save(output_file)
            logger.info("成功生成测试音频")
            return
        except Exception as test_e:
            logger.error(f"生成测试音频也失败: {str(test_e)}")
            raise Exception(f"使用语音 {self.voice} 进行语音合成失败，已尝试 {max_retries} 次，无法生成测试音频")


class DoubaoTTS(TTS):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.appid = config.get("appid")
        self.access_token = config.get("access_token")
        self.cluster = config.get("cluster")
        self.voice = config.get("voice")

        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.header = {"Authorization": f"Bearer;{self.access_token}"}

    def generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    async def text_to_speak(self, text, output_file):
        request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": self.cluster
            },
            "user": {
                "uid": "1"
            },
            "audio": {
                "voice_type": self.voice,
                "encoding": "wav",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        resp = requests.post(self.api_url, json.dumps(request_json), headers=self.header)
        if "data" in resp.json():
            data = resp.json()["data"]
            file_to_save = open(output_file, "wb")
            file_to_save.write(base64.b64decode(data))


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls_map = {
        "DoubaoTTS": DoubaoTTS,
        "EdgeTTS": EdgeTTS,
        "FallbackTTS": FallbackTTS,
        # 可扩展其他TTS实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的TTS类型: {class_name}")

class FallbackTTS(TTS):
    """带回退机制的TTS类，当主TTS失败时使用备选TTS"""
    def __init__(self, config, delete_audio_file):
        # 获取主TTS和备选TTS的配置
        primary_tts_config = config.get("primary", {})
        fallback_tts_config = config.get("fallback", {})

        # 初始化主TTS和备选TTS
        self.primary_tts = create_instance(primary_tts_config.get("type", "EdgeTTS"), primary_tts_config, delete_audio_file)
        self.fallback_tts = create_instance(fallback_tts_config.get("type", "DoubaoTTS"), fallback_tts_config, delete_audio_file)
        self.delete_audio_file = delete_audio_file
        self.output_file = self.primary_tts.output_file

    def generate_filename(self, extension=".mp3"):
        return self.primary_tts.generate_filename(extension)

    async def text_to_speak(self, text, output_file):
        try:
            # 首先尝试使用主TTS
            await self.primary_tts.text_to_speak(text, output_file)
            return
        except Exception as primary_error:
            logger.warning(f"主TTS失败: {str(primary_error)}，尝试使用备选TTS")
            try:
                # 主TTS失败，使用备选TTS
                await self.fallback_tts.text_to_speak(text, output_file)
                return
            except Exception as fallback_error:
                logger.error(f"备选TTS也失败: {str(fallback_error)}")
                raise Exception(f"所有TTS尝试均失败。主TTS错误: {str(primary_error)}，备选TTS错误: {str(fallback_error)}")


# if __name__ == "__main__":
#     config = read_config(get_project_dir() + "config.yaml")
#     tts = create_instance(
#         config["selected_module"]["TTS"],
#         config["TTS"][config["selected_module"]["TTS"]],
#         config["delete_audio"]
#     )
#     tts.output_file = get_project_dir() + tts.output_file
#     file_path = tts.to_tts("你好，测试")
#     print(file_path)
#     print(tts.wav_to_opus_data(file_path))
if __name__ == "__main__":
    config = read_config(get_project_dir() + "config.yaml")
    tts = create_instance(
        config["selected_module"]["TTS"],
        config["TTS"][config["selected_module"]["TTS"]],
        config["delete_audio"]
    )
    tts.output_file = get_project_dir() + tts.output_file
    file_path = tts.to_tts("你好，测试")
    if file_path:
        print(file_path)
        print(tts.wav_to_opus_data(file_path))
    else:
        print("TTS生成失败，无法继续处理")


