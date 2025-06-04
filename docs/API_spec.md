# 服务调用图



~~~mermaid
flowchart TB
  %% 客户端与网关
  subgraph 客户端
    A[Client App]
  end

  subgraph 网关层 [API Gateway]
    G1["/auth/login<br/>/auth/refresh<br/>/auth/logout"]
    G2["/asr/batch (POST)<br/>/asr/stream (WS)"]
    G3["/analysis/result (POST)"]
  end

  %% 后端服务模块
  subgraph ASR [service-asr]
    ASR1[噪声抑制]
    ASR2[多说话人分离]
  end

  subgraph NLP [service-nlp]
    NLP1[文本清洗]
    NLP2[分词]
    NLP3[意图识别]
    NLP4[关键词提取]
  end

  subgraph Analysis [service-analysis]
    AN1[情感分类]
    AN2[情感强度评估]
    AN3[多轮对话跟踪]
  end

  subgraph Feedback [service-feedback]
    FB1[WebSocket 推送]
    FB2[纠错接口]
    FB3[满意度反馈]
  end

  %% 流程
  A -->|POST /auth/login<br/>或 流式 ASR| G1 --> G2
  G2 -->|上传音频| ASR1 --> ASR2 -->|转写文本| NLP1 --> NLP2 --> NLP3 --> NLP4 -->|文本分析请求| AN1 --> AN2 --> AN3 -->|分析结果| G3 --> A

  %% 反馈
  AN3 -->|发送分析事件| FB1
  A -->|WS /feedback/ws| FB1
  A -->|POST /feedback/correction| FB2
  A -->|POST /feedback/satisfaction| FB3

~~~



# API Specification

本文档列出系统各个服务模块的对外 HTTP / WebSocket 接口，包括路径、方法、请求和响应示例，以及状态码说明。

### 1. 接入层（Ingress）

**Ingress（API Gateway）** 暴露给外部客户端，也把请求路由到内部服务；

#### 1.1 API 网关 (gateway)

所有下列接口均以 `https://{gateway-host}` 为前缀。网关主要负责鉴权、限流、日志埋点、流量转发。

| 接口       | 方法 | 路径            | 描述                                       |
| ---------- | ---- | --------------- | ------------------------------------------ |
| 登录       | POST | `/auth/login`   | 用户认证，返回 accessToken 和 refreshToken |
| 刷新 Token | POST | `/auth/refresh` | 刷新并颁发新的 accessToken / refreshToken  |
| 登出       | POST | `/auth/logout`  | 注销当前会话                               |

**请求 / 响应示例**：

```
POST /auth/login HTTP/1.1
Content-Type: application/json

{ "username": "alice", "password": "secret" }
HTTP/1.1 200 OK
Content-Type: application/json

{
  "accessToken": "eyJ...",
  "refreshToken": "def...",
  "expiresIn": 3600
}
```

### 2. 语音转文字服务（service-asr）

服务地址：`http://{asr-host}`

| 接口     | 方法             | 路径          | 描述                           |
| -------- | ---------------- | ------------- | ------------------------------ |
| 批量转写 | POST             | `/asr/batch`  | 接收音频文件列表，返回转写结果 |
| 流式转写 | WebSocket 或 SSE | `/asr/stream` | 接收音频流，实时返回转写片段   |

**请求 / 响应示例**：

```
POST /asr/batch HTTP/1.1
Content-Type: multipart/form-data

files: [audio1.wav, audio2.wav]
[
  { "file": "audio1.wav", "transcript": "Hello world", "segments": [...] },
  { "file": "audio2.wav", "transcript": "你好，世界", "segments": [...] }
]
```

### 3. NLP 分析服务（service-nlp）

服务地址：`http://{nlp-host}`

| 接口            | 方法 | 路径            | 描述                           |
| --------------- | ---- | --------------- | ------------------------------ |
| 文本清洗 & 分词 | POST | `/nlp/tokenize` | 清洗文本并分词                 |
| 意图识别        | POST | `/nlp/intent`   | 基于 BERT Fine-tune 的意图分类 |
| 关键词提取      | POST | `/nlp/keyword`  | TF-IDF / RAKE 提取关键词       |

**请求 / 响应示例**：

```
POST /nlp/intent HTTP/1.1
Content-Type: application/json

{ "text": "我想查询余额" }
{ "intent": "query_balance", "confidence": 0.95 }
```

### 4. 情感与意图分析服务（service-analysis）

服务地址：`http://{analysis-host}`

| 接口     | 方法 | 路径                  | 描述           |
| -------- | ---- | --------------------- | -------------- |
| 情感分类 | POST | `/analysis/sentiment` | 多分类情感分析 |
| 强度评估 | POST | `/analysis/intensity` | 情感强度回归   |
| 对话跟踪 | POST | `/analysis/dialog`    | 多轮上下文管理 |

**请求 / 响应示例**：

```
POST /analysis/sentiment HTTP/1.1
Content-Type: application/json

{ "text": "这个产品太棒了" }
{ "sentiment": "positive", "score": 0.87 }
```

### 5. 客服反馈模块（service-feedback）

服务地址：`http://{feedback-host}`

| 接口       | 方法      | 路径                     | 描述                         |
| ---------- | --------- | ------------------------ | ---------------------------- |
| 推送连接   | WebSocket | `/feedback/ws`           | 客户端通过 WS 建立实时推送   |
| 提交纠错   | POST      | `/feedback/correction`   | 用户对转写或分析结果进行纠错 |
| 提交满意度 | POST      | `/feedback/satisfaction` | 用户打分、留言               |

**请求 / 响应示例**：

```
POST /feedback/correction HTTP/1.1
Content-Type: application/json
Authorization: Bearer <token>

{ "messageId": "12345", "correctedText": "正确文本" }
HTTP/1.1 204 No Content
```

### 6. 存储层说明

存储层不提供对外 HTTP 接口，由各服务内部通过标准驱动访问：

- PostgreSQL 存储对话、用户、反馈等结构化数据
- MongoDB 存储半结构化日志与向量数据
- MinIO 对象存储音频文件及媒体资源

> 文档自动生成于项目结构变更后，请根据实际部署地址和安全规范进行补充和调整。