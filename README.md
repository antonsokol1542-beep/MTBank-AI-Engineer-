# MTBank — Speech Analytics for Contact Center

> **AI Engineer — Test Assignment**  
> Stack: Python 3.11 · FastAPI · faster-whisper · LangGraph · OpenWebUI Pipelines · Docker Compose

## Quick Start (English)

```bash
cp .env.example .env          # add OPENAI_API_KEY (OpenRouter / Groq / OpenAI)
docker compose up -d          # starts OpenWebUI :3000, Pipelines :9099, API :8000
```

Open **<http://localhost:3000>**, select the **MTBank Analytics** pipeline, upload a WAV/MP3/OGG file — get a full call analysis report in the chat.

**REST API** — `POST /analyze` (multipart `file=<audio>` or form field `url`):

```json
{
  "transcript": [{"speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "..."}],
  "classification": {"topic": "кредиты", "priority": "medium"},
  "quality_score": {"total": 78, "checklist": {"greeting": true, "need_detection": true, "solution_provided": true, "farewell": false}},
  "compliance": {"passed": true, "issues": []},
  "summary": "Клиент обратился...",
  "action_items": ["Отправить КП на email клиента"]
}
```

Swagger UI: **<http://localhost:8000/docs>**

### Architecture

```
User (browser)
  └─► OpenWebUI :3000
        └─► OpenWebUI Pipelines :9099  (pipeline.py / analytics_pipeline.py)
              └─► FastAPI :8000  ──► faster-whisper (ASR + diarization)
                                 └─► LangGraph agents
                                       ├─ classifier
                                       ├─ quality_agent  ┐ parallel
                                       ├─ compliance     ┘
                                       └─ summarizer
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **LangGraph** | Explicit state graph, easy to trace each agent step, parallel fan-out with `add_edge(classifier → quality)` + `add_edge(classifier → compliance)` |
| **Pipelines → FastAPI split** | Pipelines handle OpenWebUI protocol; FastAPI exposes a clean REST API reusable from CLI, tests, and other clients |
| **faster-whisper medium / int8** | Best CPU speed/accuracy tradeoff; `beam_size=1` + `vad_filter` keeps latency < 60 s for files up to 5 min |
| **Groq / llama-3.3-70b-versatile** | Near-zero cost, OpenAI-compatible, fast inference; any OpenAI-compatible endpoint works via `OPENAI_API_BASE_URL` |

---

## MTBank — Речевая аналитика контакт-центра

> **AI Engineer — тестовое задание**  
> Стек: Python 3.11 · FastAPI · faster-whisper · LangGraph · OpenWebUI Pipelines · Docker Compose

---

## Архитектурная схема

```
┌────────────────────────────────────────────────────────────────────┐
│               OpenWebUI (port 3000) — чат-интерфейс               │
│  Пользователь загружает аудио → получает Markdown-отчёт в чате    │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ OpenAI-compatible API
┌──────────────────────────▼─────────────────────────────────────────┐
│          OpenWebUI Pipelines (port 9099)                           │
│  pipeline.py ─────────────────────── analytics_pipeline.py        │
│  (единый Pipeline по скелету ТЗ)      (расширенный вариант)       │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ HTTP POST /analyze
┌──────────────────────────▼─────────────────────────────────────────┐
│              FastAPI Backend (port 8000)                           │
│                                                                    │
│  ┌─────────────────┐   ┌────────────────────────────────────────┐  │
│  │   ASR Service   │   │         LangGraph Pipeline             │  │
│  │                 │   │                                        │  │
│  │ faster-whisper  │   │  START                                 │  │
│  │ (medium model+) │   │    ↓                                   │  │
│  │                 │   │  🏷️ classifier                         │  │
│  │ + pyannote      │   │   ↙ ↘  (параллельно)                  │  │
│  │   diarization   │   │  ⭐quality  🛡️compliance               │  │
│  │   (или fallback)│   │   ↘ ↙  (fan-in)                       │  │
│  └─────────────────┘   │  📝 summarizer                         │  │
│                        │    ↓                                   │  │
│                        │  END                                   │  │
│                        └────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Обоснование архитектурных решений

| Решение | Обоснование |
|---|---|
| **LangGraph** | Явный граф состояний с возможностью трассировки каждого шага. Легко добавить новых агентов или переключиться на параллельное выполнение (`Send()`). Альтернативы (custom Supervisor) потребовали бы больше boilerplate-кода без выигрыша в гибкости |
| **Параллельный граф** | Classifier → (Quality ‖ Compliance) → Summarizer: quality и compliance работают параллельно (fan-out/fan-in в LangGraph), что сокращает время обработки на ~10–15 с. Summarizer видит результаты обоих агентов |
| **faster-whisper medium** | Оптимальный баланс скорость/точность для CPU. На GPU рекомендуется `large-v3` |
| **Pipeline → API** | Pipelines вызывают FastAPI-backend — разделение ответственности, переиспользование логики из CLI и API |
| **Groq / llama-3.3-70b-versatile** | Быстрый и бесплатный inference, OpenAI-совместимый API. Любой провайдер меняется через переменные `OPENAI_API_KEY` + `OPENAI_API_BASE_URL` + `LLM_MODEL` |

---

## Быстрый старт

### 1. Требования

- Docker и Docker Compose
- OpenAI API key (или совместимый endpoint)

### 2. Настройка

```bash
cp .env.example .env
# Отредактируйте .env: добавьте OPENAI_API_KEY
```

### 3. Запуск

```bash
docker compose up -d
```

Сервисы:

| Сервис | URL |
|---|---|
| OpenWebUI (чат) | http://localhost:3000 |
| Pipelines API | http://localhost:9099 |
| FastAPI Backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

### 4. Подключение пайплайна в OpenWebUI

1. Admin Panel → Connections → добавить `http://pipelines:9099` (API Key из `.env`)
2. Пайплайны подгрузятся автоматически из `./pipelines/`
3. В новом чате выберите модель **MTBank Analytics**

---

## API

### `POST /analyze`

Основной эндпоинт (по ТЗ). Полный анализ звонка.

**Request** (multipart/form-data):
```
file=<audio.wav>   # WAV/MP3/OGG/FLAC
# или
url=https://...    # URL до аудиофайла
```

**Response**:
```json
{
  "audio_duration": 34.5,
  "transcript": [
    { "speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "Добрый день, МТБанк, меня зовут Анна." },
    { "speaker": "Клиент",   "start": 4.5, "end": 8.1, "text": "Здравствуйте, хочу узнать про кредит." }
  ],
  "classification": { "topic": "Потребительский кредит", "priority": "medium" },
  "quality_score": {
    "total": 78,
    "checklist": {
      "greeting": true,
      "need_detection": true,
      "solution_provided": true,
      "farewell": false
    },
    "issues": ["Оператор не попрощался с клиентом"]
  },
  "compliance": { "passed": true, "issues": [] },
  "summary": "Клиент обратился по вопросу оформления потребительского кредита...",
  "action_items": ["[Оператор] Отправить клиенту инструкцию на email"],
  "processing_time": 22.4
}
```

### `POST /api/v1/transcribe`

Только транскрипция (ASR + диаризация), без агентов.

---

## Тестовые данные

### Генерация аудиофайлов

```bash
pip install edge-tts
python scripts/generate_audio.py
```

Создаёт 5 WAV-файлов в `test_data/`:

| Файл | Описание | Длительность | Sample Rate |
|---|---|---|---|
| `credit_loan.wav` | Кредит наличными (2 спикера) | ~3 мин | 16kHz |
| `card_issue.wav` | Блокировка карты | ~1 мин | **8kHz** (телефония) |
| `complaint.wav` | Жалоба клиента | ~1 мин | 16kHz |
| `mortgage.wav` | Ипотека (2 спикера) | ~2 мин | 16kHz |
| `account_info.wav` | Баланс и уведомления | ~1 мин | 16kHz |

### WER (Word Error Rate)

```bash
pip install jiwer
python scripts/calculate_wer.py
python scripts/calculate_wer.py --model large-v3
```

| Файл | Длительность | WER (medium) | WER (large-v3) | Примечание |
|---|---|---|---|---|
| credit_loan.wav | 120 с | **9.8%** | — | TTS, диалог 2 спикера |
| card_issue.wav | 82 с | **5.3%** | — | 8kHz телефония |
| complaint.wav | 99 с | **6.2%** | — | TTS, 2 спикера |
| mortgage.wav | 117 с | **11.9%** | — | TTS, диалог 2 спикера |
| account_info.wav | 85 с | **14.4%** | — | TTS, цифровые коды |
| **Итого / среднее** | **503 с (8.4 мин)** | **9.5%** | — | |

Замеры сделаны на CPU (faster-whisper medium, int8, язык: ru).
WER large-v3 ожидаемо ниже на ~30–40%, для точных цифр требуется GPU-сервер.

> **Интерпретация:** 9.5% WER на синтетической TTS-речи — ожидаемо, т.к. TTS-голос чище реального. На реальных телефонных записях (шум АТС, акценты) medium даёт 15–30% WER.

---

## Тесты

```bash
# В контейнере
docker compose exec api pytest tests/ -v

# Локально (из корня репозитория)
cd api && pip install -r requirements.txt && cd ..
pytest tests/ -v
```

Покрытие:

| Файл | Что тестируется |
|---|---|
| `tests/unit/test_agents.py` | Юнит-тесты каждого из 4 агентов (mock LLM) |
| `tests/unit/test_asr.py` | Диаризация, выравнивание спикеров, граничные случаи |
| `tests/integration/test_api.py` | Все эндпоинты API, структура JSON-ответа, коды ошибок |

---

## Диаризация

| Режим | Условие | Метод |
|---|---|---|
| **Точный** | `HF_TOKEN` задан | `pyannote/speaker-diarization-3.1` |
| **Эвристика** | Без токена | Смена спикера при паузе > 0.3 с или если один спикер говорит > 6 с без пауз (первый = Оператор) |

---

## Структура репозитория

> **Отклонение от рекомендованной структуры ТЗ:** ТЗ предлагает разместить `agents/` и `asr/` в корне репозитория. В данном решении они находятся внутри `api/` — это осознанное архитектурное решение: агенты и ASR-сервис запускаются исключительно в контексте FastAPI-бэкенда (свой virtualenv, Dockerfile, зависимости). Вынос их в корень создал бы ложную иллюзию, что они могут использоваться независимо от бэкенда. `pipeline.py` и `pipelines/` остаются в корне, т.к. они исполняются в контейнере OpenWebUI Pipelines — отдельном окружении.

```
.
├── pipeline.py                    # Основной OpenWebUI Pipeline (по ТЗ)
├── docker-compose.yml
├── .env.example
├── pipelines/
│   ├── asr_pipeline.py            # Pipeline: только транскрипция
│   └── analytics_pipeline.py     # Pipeline: полный анализ (расширенный вид)
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                    # FastAPI: POST /analyze, /api/v1/transcribe
│   ├── config.py
│   ├── agents/
│   │   ├── classifier.py          # 🏷️ Классификатор темы и приоритета
│   │   ├── quality_agent.py       # ⭐ Агент качества оператора
│   │   ├── compliance_agent.py    # 🛡️ Комплаенс-агент
│   │   └── summarizer.py          # 📝 Суммаризатор
│   ├── services/
│   │   └── asr.py                 # faster-whisper + диаризация
│   ├── models/
│   │   └── schemas.py             # Pydantic-схемы запросов/ответов
│   ├── orchestrator/
│   │   └── graph.py               # LangGraph pipeline
│   └── utils/
│       └── logging.py             # JSON-логирование (structlog)
├── test_data/                     # Аудиофайлы + эталонные транскрипты
│   ├── credit_loan.wav/txt
│   ├── card_issue.wav/txt         # 8kHz телефония
│   ├── complaint.wav/txt
│   ├── mortgage.wav/txt
│   └── account_info.wav/txt
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_agents.py
│   │   └── test_asr.py
│   └── integration/
│       └── test_api.py
└── scripts/
    ├── generate_audio.py           # Генерация тестовых аудио (edge-tts)
    └── calculate_wer.py            # Расчёт WER (jiwer)
```
