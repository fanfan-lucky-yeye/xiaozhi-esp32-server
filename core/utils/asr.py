import time
import wave
import os
from abc import ABC, abstractmethod
import logging
from typing import Optional, Tuple, List
import uuid

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = logging.getLogger(__name__)


class ASR(ABC):
    @abstractmethod
    def save_audio_to_file(self, audio_data: List[bytes], session_id: str) -> str:
        """将音频数据保存为WAV文件"""
        pass

    @abstractmethod
    def speech_to_text(self, audio_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        pass


class FunASR(ASR):
    def __init__(self, config: dict, delete_audio_file: bool):
        self.model_dir = config.get("model_dir")
        self.output_dir = config.get("output_dir")  # 修正配置键名
        self.delete_audio_file = delete_audio_file

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 检查是否使用本地模型
        local_model = config.get("local_model", False)
        
        self.model = AutoModel(
            model=self.model_dir if local_model else "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_kwargs={"max_single_segment_time": 30000},
            disable_update=local_model,  # 如果使用本地模型，则禁用更新
            hub="hf" if local_model else "ms",
            # device="cuda:0",  # 启用GPU加速
        )

    def save_audio_to_file(self, audio_data: List[bytes], session_id: str) -> str:
        """将音频数据保存为WAV文件（支持原始PCM数据）"""
        file_name = f"asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)

        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes = 16-bit
            wf.setframerate(16000)
            wf.writeframes(b"".join(audio_data))

        return file_path

    def speech_to_text(self, audio_data: List[bytes], session_id: str) -> Tuple[Optional[str], Optional[str]]:
        """语音转文本主处理逻辑"""
        file_path = None
        try:
            # 保存音频文件
            start_time = time.time()
            file_path = self.save_audio_to_file(audio_data, session_id)
            logger.debug(f"音频文件保存耗时: {time.time() - start_time:.3f}s | 路径: {file_path}")

            # 语音识别
            start_time = time.time()
            result = self.model.generate(
                input=file_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            text = rich_transcription_postprocess(result[0]["text"])
            logger.debug(f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {text}")

            return text, file_path

        except Exception as e:
            logger.error(f"语音识别失败: {e}", exc_info=True)
            return None, None

        finally:
            # 文件清理逻辑
            if self.delete_audio_file and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"已删除临时音频文件: {file_path}")
                except Exception as e:
                    logger.error(f"文件删除失败: {file_path} | 错误: {e}")


def create_instance(class_name: str, *args, **kwargs) -> ASR:
    """工厂方法创建ASR实例"""
    cls_map = {
        "FunASR": FunASR,
        # 可扩展其他ASR实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的ASR类型: {class_name}")