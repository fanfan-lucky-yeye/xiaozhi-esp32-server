"""
语音问答服务 API
支持文本/语音输入，返回文本/语音输出
"""
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi

# 配置日志
from config.logger import setup_logging
from config.settings import load_config

# 核心模块
from core.utils import asr, llm, tts
from core.utils.dialogue import Message, Dialogue
from core.utils.util import get_string_no_punctuation_or_emoji

setup_logging()
config = load_config()

# 初始化核心模块实例
_vad, _asr, _llm, _tts = None, None, None, None



def init_modules():
    """初始化核心模块"""
    global _asr, _llm, _tts
    _asr = asr.create_instance(
        config["selected_module"]["ASR"],
        config["ASR"][config["selected_module"]["ASR"]],
        config["delete_audio"]
    )
    _llm = llm.create_instance(
        config["selected_module"]["LLM"],
        config["LLM"][config["selected_module"]["LLM"]],
    )
    _tts = tts.create_instance(
        config["selected_module"]["TTS"],
        config["TTS"][config["selected_module"]["TTS"]],
        config["delete_audio"]
    )
    # 设置TTS输出目录
    if hasattr(_tts, 'output_file'):
        _tts.output_file = os.path.join(os.path.dirname(__file__), _tts.output_file)
    elif hasattr(_tts, 'primary_tts'):
        _tts.output_file = os.path.join(os.path.dirname(__file__), _tts.primary_tts.output_file)

init_modules()

# 创建FastAPI应用
app = FastAPI(
    title="语音问答服务 API",
    description="支持文本/语音输入，返回文本/语音输出的智能问答服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 会话存储（内存方式，生产环境建议使用Redis）
sessions: Dict[str, Dialogue] = {}

def get_prompt() -> str:
    """获取系统提示词"""
    prompt = config.get("prompt", "")
    if "{date_time}" in prompt:
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = prompt.replace("{date_time}", date_time)
    return prompt

def get_session_dialogue(session_id: str) -> Dialogue:
    """获取或创建会话对话"""
    if session_id not in sessions:
        dialogue = Dialogue()
        dialogue.put(Message(role="system", content=get_prompt()))
        sessions[session_id] = dialogue
    return sessions[session_id]

def generate_session_id() -> str:
    """生成会话ID"""
    return str(uuid.uuid4())


@app.post("/ask", summary="智能问答接口", 
          description="支持文本和语音两种输入方式，返回文本和语音输出")
async def ask(
    request: Request,
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    return_audio: Optional[bool] = Form(True)
) -> Dict[str, Any]:
    # 尝试从 JSON 请求体获取参数
    try:
        body = await request.json()
        if 'text' in body and not text:
            text = body.get('text')
        if 'session_id' in body and not session_id:
            session_id = body.get('session_id')
        if 'return_audio' in body:
            return_audio = body.get('return_audio')
    except:
        pass
    """
    智能问答接口
    
    **参数说明**：
    - `text`: 文本输入（与audio二选一）
    - `audio`: 语音文件输入（支持WAV/MP3格式，与text二选一）
    - `session_id`: 会话ID，用于保持对话上下文（可选，不传则自动生成）
    - `return_audio`: 是否返回语音文件（默认True）
    
    **返回值**：
    - `session_id`: 会话ID
    - `input_text`: 识别/输入的文本
    - `response_text`: 回答文本
    - `audio_url`: 语音文件URL（当return_audio为True时）
    - `audio_duration`: 语音时长（秒）
    """
    # 验证输入
    if not text and not audio:
        raise HTTPException(status_code=400, detail="必须提供text或audio参数")
    
    # 生成或获取会话ID
    if not session_id:
        session_id = generate_session_id()
    
    dialogue = get_session_dialogue(session_id)
    
    # 处理语音输入
    input_text = text
    if audio:
        # 获取文件扩展名
        file_ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else '.wav'
        
        # 保存上传的音频文件
        temp_path = os.path.join(config["ASR"]["FunASR"]["output_dir"], f"upload_{uuid.uuid4()}{file_ext}")
        wav_path = os.path.join(config["ASR"]["FunASR"]["output_dir"], f"upload_{uuid.uuid4()}.wav")
        
        with open(temp_path, "wb") as f:
            f.write(await audio.read())
        
        # 使用ASR识别语音
        try:
            # 如果不是WAV格式，转换为WAV
            if file_ext != '.wav':
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_file(temp_path)
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                audio_segment.export(wav_path, format='wav')
            else:
                import shutil
                shutil.copy(temp_path, wav_path)
            
            # 读取WAV文件
            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio_data = [frames]
            
            input_text, _ = _asr.speech_to_text(audio_data, session_id)
            
            # 清理上传文件
            for path in [temp_path, wav_path]:
                if os.path.exists(path):
                    os.remove(path)
        except Exception as e:
            # 清理上传文件
            for path in [temp_path, wav_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")
    
    if not input_text or len(input_text.strip()) == 0:
        raise HTTPException(status_code=400, detail="无法识别语音内容或文本为空")
    
    # 获取干净的文本（去除首尾标点和表情）
    clean_text = get_string_no_punctuation_or_emoji(input_text)
    
    # 调用LLM生成回答
    dialogue.put(Message(role="user", content=clean_text))
    
    try:
        llm_responses = _llm.response(None, dialogue.get_llm_dialogue())
        response_text = "".join([r for r in llm_responses])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")
    
    # 保存对话历史
    dialogue.put(Message(role="assistant", content=response_text))
    
    # 生成语音（如果需要）
    audio_url = None
    audio_duration = 0

    if return_audio and response_text:
        try:
            print(f"[DEBUG] 开始生成语音，文本: {response_text[:20]}...")
            tts_file = await _tts.to_tts(response_text)
            print(f"[DEBUG] TTS返回文件路径: {tts_file}")
            
            if tts_file:
                # 检查文件是否存在
                if os.path.exists(tts_file):
                    print(f"[DEBUG] 语音文件存在: {tts_file}")
                    # 获取文件名作为URL路径
                    audio_filename = os.path.basename(tts_file)
                    audio_url = f"/audio/{audio_filename}"
                    
                    # 获取音频时长
                    audio_duration = _tts.get_audio_duration(tts_file)
                    print(f"[DEBUG] 音频时长: {audio_duration}")
                else:
                    print(f"[DEBUG] 语音文件不存在: {tts_file}")
            else:
                print("[DEBUG] TTS返回None，语音生成失败")

                # 保留语音文件供下载，不立即删除
                # 语音文件会在后续请求或定时任务中清理
        except Exception as e:
            print(f"[DEBUG] TTS异常: {e}")
            # TTS失败不影响文本响应
            pass
        
    return {
        "session_id": session_id,
        "input_text": input_text,
        "response_text": response_text,
        "audio_url": audio_url,
        "audio_duration": audio_duration
    }

@app.get("/audio/{filename}", summary="获取语音文件")
async def get_audio(filename: str):
    """
    获取生成的语音文件
    
    **参数说明**：
    - `filename`: 语音文件名
    
    **返回值**：
    - 语音文件流（MP3格式）
    """
    # 使用项目根目录下的 tmp 文件夹作为语音文件存储路径
    audio_path = os.path.join(os.path.dirname(__file__), "tmp", filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="语音文件不存在")
    
    # 设置完整的响应头，确保浏览器和Postman能正确识别
    headers = {
        "Content-Type": "audio/mpeg",
        "Content-Disposition": f"inline; filename=\"{filename}\"",
        "Access-Control-Allow-Origin": "*"
    }
    
    return FileResponse(audio_path, headers=headers)

@app.delete("/session/{session_id}", summary="删除会话")
async def delete_session(session_id: str):
    """
    删除指定会话，清除对话历史
    
    **参数说明**：
    - `session_id`: 会话ID
    
    **返回值**：
    - `success`: 是否成功
    """
    if session_id in sessions:
        del sessions[session_id]
        return {"success": True, "message": "会话已删除"}
    return {"success": False, "message": "会话不存在"}

@app.get("/session/{session_id}", summary="获取会话历史")
async def get_session_history(session_id: str):
    """
    获取指定会话的对话历史
    
    **参数说明**：
    - `session_id`: 会话ID
    
    **返回值**：
    - `dialogue`: 对话历史列表
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    dialogue = sessions[session_id]
    return {"session_id": session_id, "dialogue": dialogue.get_llm_dialogue()}

@app.get("/health", summary="健康检查")
async def health_check():
    """
    服务健康检查接口
    """
    return {"status": "ok", "service": "xiaozhi-api", "version": "1.0.0"}

# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="语音问答服务 API",
        version="1.0.0",
        description="支持文本/语音输入，返回文本/语音输出的智能问答服务\n\n"
                    "**使用说明**：\n"
                    "- POST /ask 进行问答，支持文本和语音输入\n"
                    "- GET /audio/{filename} 获取生成的语音文件\n"
                    "- DELETE /session/{session_id} 删除会话\n"
                    "- GET /session/{session_id} 获取会话历史",
        routes=app.routes,
    )
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    server_config = config.get("server", {"ip": "0.0.0.0", "port": 8000})
    print(f"服务启动: http://{server_config['ip']}:{server_config['port']}")
    print(f"API文档: http://{server_config['ip']}:{server_config['port']}/docs")
    # 禁用自动重载以避免文件变化循环检测
    uvicorn.run(
        "main:app", 
        host=server_config["ip"], 
        port=server_config["port"], 
        reload=False
    )