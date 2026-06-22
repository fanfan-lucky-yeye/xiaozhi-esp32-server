
### 🚀 快速开始

#### 环境要求

- Python 3.9+
- 内存：建议 8GB+
- 磁盘：至少 5GB 可用空间（模型文件）

## 📥 模型文件下载

由于 GitHub 文件大小限制，大型模型文件未包含在仓库中。请手动下载：

### SenseVoiceSmall 模型
- 下载地址：https://github.com/modelscope/FunASR
- 放置路径：`models/SenseVoiceSmall/model.pt`

### Silero VAD 模型
- 下载地址：https://github.com/snakers4/silero-vad
- 放置路径：`models/snakers4-silero-vad/data/`


#### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd xiaozhi-esp32-server

# 2. 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
# 编辑 config.yaml，填入你的 LLM API Key

# 5. 启动服务
python main.py
```

#### 配置说明

`config.yaml` 主要配置项：

```yaml
server:
  ip: 0.0.0.0
  port: 8000

selected_module:
  ASR: FunASR
  LLM: ChatGLMLLM
  TTS: FallbackTTS

LLM:
  ChatGLMLLM:
    api_key: your-api-key
```

### 🔌 API 接口文档

#### 1. 智能问答接口

**POST** `/ask`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | string | 否 | 文本输入（与audio二选一） |
| audio | file | 否 | 语音文件（WAV/MP3） |
| session_id | string | 否 | 会话ID |
| return_audio | bool | 否 | 是否返回语音（默认true） |

**响应示例：**
```json
{
    "session_id": "xxx",
    "input_text": "你好",
    "response_text": "你好！有什么可以帮助你的？",
    "audio_url": "/audio/tts-xxx.mp3",
    "audio_duration": 2.0
}
```

#### 2. 获取语音文件

**GET** `/audio/{filename}`

#### 3. 会话管理

**GET** `/session/{session_id}` - 获取会话历史

**DELETE** `/session/{session_id}` - 删除会话

#### 4. 健康检查

**GET** `/health`

### 🐳 Docker 部署

```bash
# 构建镜像
docker build -t xiaozhi-api .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/tmp:/app/tmp xiaozhi-api
```

### 📝 Postman 使用示例

#### 文本问答
- Method: POST
- URL: `http://localhost:8000/ask`
- Body: JSON
```json
{"text": "你好", "return_audio": true}
```

#### 语音问答
- Method: POST
- URL: `http://localhost:8000/ask`
- Body: form-data
- audio: 选择语音文件
- return_audio: true

### 🔧 故障排除

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 模型下载失败 | 网络问题 | 检查网络或手动下载模型 |
| API Key 无效 | 密钥错误 | 检查 config.yaml 配置 |
| 端口占用 | 8000端口被占用 | 修改 config.yaml 中的端口 |

### 📁 项目结构
xiaozhi-esp32-server/
├── main.py                 # FastAPI入口 ✅
├── config.yaml             # 配置文件 ✅
├── requirements.txt        # 依赖列表 ✅
├── config/
│   ├── logger.py          # 日志配置 ✅
│   └── settings.py        # 配置加载 ✅
├── core/utils/
│   ├── asr.py             # ASR模块 ✅
│   ├── tts.py             # TTS模块 ✅
│   ├── llm.py             # LLM模块 ✅
│   ├── dialogue.py        # 会话管理 ✅
│   ├── vad.py             # VAD模块 ✅
│   └── util.py            # 工具函数 ✅
├── models/                # 模型文件 ✅
└── tmp/                   # 临时文件目录 ✅

## 📥 模型文件下载说明

由于 GitHub 文件大小限制，以下大型模型文件**未包含**在仓库中，请手动下载并放到对应目录：

### 1. SenseVoiceSmall ASR 模型
- **文件：** `model.pt`
- **大小：** 约 50MB
- **放置路径：** `models/SenseVoiceSmall/model.pt`
- **下载方式：** 首次启动 `python main.py` 时，FunASR 会自动从 ModelScope 下载
- **手动下载地址：** https://www.modelscope.cn/models/iic/SenseVoiceSmall

### 2. Silero VAD 模型
- **文件：** `data/*.onnx`, `data/*.jit`
- **放置路径：** `models/snakers4_silero-vad/data/`
- **下载地址：** https://github.com/snakers4/silero-vad/tree/master/files

### 3. 示例音频（可选）
- 放置路径：`models/SenseVoiceSmall/example/zh.mp3`
- 用于快速测试语音识别功能

### ✅ 首次启动步骤

1. 克隆或下载本仓库
2. 按上述说明放置模型文件
3. 安装依赖：`pip install -r requirements.txt`
4. 编辑 `config.yaml`，填入 LLM API Key
5. 启动服务：`python main.py`
6. 访问 http://localhost:8000/docs 查看 API 文档
