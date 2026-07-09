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

### Which LLM to use?

Any **OpenAI-compatible** endpoint works — swap `OPENAI_API_BASE_URL` + `LLM_MODEL` in `.env`, no code changes needed.

| Provider | Free tier | Latency | Notes |
| --- | --- | --- | --- |
| **Groq** (`api.groq.com`) | ✅ 30 rpm free | ~1–2 s | Recommended — fastest inference, reliable free tier |
| **OpenRouter** (`openrouter.ai`) | ✅ many free models | 2–5 s | Wide model selection; free models have rate limits |
| **Together AI** (`api.together.ai`) | ✅ $1 credit | 2–4 s | Good for llama/mistral variants |
| **OpenAI** | ❌ paid | ~2 s | `gpt-4o-mini` works well, ~$0.01 per call |
| Local (Ollama) | ✅ free | 10–60 s | Requires GPU for acceptable speed |

**Why Groq by default:** no GPU required on the server side, free tier covers hundreds of calls per day, and llama-3.3-70b produces reliable structured JSON in Russian — which is the critical requirement for this pipeline.

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

### 5. Публичный деплой и живое демо

> **Ограничение бесплатного хостинга.** Полный стек — это 3 контейнера
> (OpenWebUI + Pipelines + FastAPI) + модель Whisper, суммарно **~3–4 ГБ RAM**.
> Бесплатные PaaS/serverless (Railway free, Render free, HF Spaces free, Vercel) дают
> ~512 МБ на сервис и не запускают многоконтейнерный ML-стек целиком — это ограничение
> бесплатной инфраструктуры под ML-нагрузку, а не архитектуры проекта. Полный стек с
> чатом OpenWebUI разворачивается **локально одной командой** `docker compose up -d`
> (шаги 1–4 выше) или на любом VM с ≥ 4 ГБ RAM.

**Живое демо REST API — Railway (бесплатно, HTTPS).**
На бесплатном хостинге публикуется FastAPI-бэкенд (`api/`, конфиг
[api/railway.json](api/railway.json)) — Swagger UI и `POST /analyze` по HTTPS:

1. Railway → **Deploy from GitHub** → выберите репозиторий
2. Service → Settings → **Root Directory = `api`**
3. Variables: `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `LLM_MODEL`,
   `WHISPER_MODEL=small`, `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_LANGUAGE=ru`
4. Демо: `https://<app>.up.railway.app/docs`

> ⚠️ На free-tier Railway (512 МБ) обязательно `WHISPER_MODEL=small` или `tiny` —
> `medium` уходит в OOM (`Killed`). `HF_TOKEN` не задавайте: без него pyannote и torch
> не грузятся (экономия ~1 ГБ), диаризация работает по эвристике.

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
| `credit_loan.wav` | Кредит наличными (2 спикера) | ~2 мин (120 с) | 16kHz |
| `card_issue.wav` | Блокировка карты (2 спикера) | ~1.5 мин (82 с) | **8kHz** (телефония) |
| `complaint.wav` | Жалоба клиента (2 спикера) | ~1.5 мин (99 с) | 16kHz |
| `mortgage.wav` | Ипотека (2 спикера) | ~2 мин (117 с) | 16kHz |
| `account_info.wav` | Баланс и уведомления (2 спикера) | ~1.5 мин (85 с) | 16kHz |

### WER (Word Error Rate)

```bash
pip install jiwer
python scripts/calculate_wer.py
python scripts/calculate_wer.py --model large-v3
```

| Файл | Длительность | Sample Rate | WER (medium) | Примечание |
|---|---|---|---|---|
| complaint.wav | 99 с | 16kHz | **6.2%** | TTS, 2 спикера |
| mortgage.wav | 117 с | 16kHz | **8.8%** | TTS, диалог 2 спикера, суммы |
| credit_loan.wav | 120 с | 16kHz | **12.1%** | TTS, диалог 2 спикера, ставки |
| account_info.wav | 85 с | 16kHz | **16.2%** | TTS, цифровые коды карт/сумм |
| card_issue.wav | 82 с | **8kHz** | **17.0%** | телефония, цифры (худший — как и ожидалось) |
| **Итого / среднее** | **503 с (8.4 мин)** | — | **12.1%** | |

Замеры сделаны на CPU (faster-whisper **medium**, `int8`, `beam_size=1`, `vad_filter`, язык `ru`) —
теми же параметрами, что и боевой ASR-сервис. Воспроизводимо: `python scripts/calculate_wer.py`.

> **Интерпретация:** средний WER 12.1% на синтетической TTS-речи. Закономерности, которые подтверждает
> подборка данных:
> - **8kHz-телефония** (`card_issue`, 17.0%) и **насыщенные цифрами** звонки (`account_info` — номера карт,
>   суммы) дают самый высокий WER — узкая полоса частот и числительные тяжелее всего для ASR.
> - **Чистая речь без цифр** (`complaint`, 6.2%) распознаётся лучше всего.
> - На реальных телефонных записях (шум АТС, акценты, перебивания) medium ожидаемо даст 15–30% WER;
>   на GPU модель `large-v3` снижает WER ещё на ~30–40%.

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
