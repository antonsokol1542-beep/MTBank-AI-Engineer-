# Тестовые данные

## Генерация аудиофайлов

```bash
pip install edge-tts
python scripts/generate_audio.py
```

Скрипт создаст 5 WAV-файлов в этой папке:

| Файл | Описание | Длительность | Sample Rate |
|---|---|---|---|
| `credit_loan.wav` | Кредит наличными (2 спикера) | ~3 мин | 16kHz |
| `card_issue.wav` | Блокировка карты | ~1 мин | **8kHz** (телефония) |
| `complaint.wav` | Жалоба клиента | ~1 мин | 16kHz |
| `mortgage.wav` | Консультация по ипотеке (2 спикера) | ~2 мин | 16kHz |
| `account_info.wav` | Баланс и уведомления | ~1 мин | 16kHz |

Эталонные транскрипты (`.txt`) уже находятся в этой папке.

## WER (Word Error Rate)

После генерации аудио и запуска системы, запустите:

```bash
pip install jiwer
python scripts/calculate_wer.py
```

| Файл | WER (medium) | WER (large-v3) | Примечание |
|---|---|---|---|
| credit_loan.wav | — | — | Синтетическая речь (edge-tts) |
| card_issue.wav | — | — | 8kHz, телефонный кодек |
| complaint.wav | — | — | Синтетическая речь |
| mortgage.wav | — | — | 2 спикера, синтетическая речь |
| account_info.wav | — | — | Синтетическая речь |

*Таблица будет заполнена после прогона на реальной системе.*
