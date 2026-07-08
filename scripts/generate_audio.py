"""
Generate Russian-language test audio files using edge-tts.

Requirements:
  pip install edge-tts asyncio

Usage:
  python scripts/generate_audio.py

Produces 5 audio files in ./audio_samples/:
  1. credit_loan.wav        — Кредит наличными (2-speaker dialogue, ~3 min)
  2. card_issue.wav         — Проблема с картой (1+ min, 8kHz telephony quality)
  3. complaint.wav          — Жалоба клиента (~1 min)
  4. mortgage.wav           — Ипотека (multi-speaker, ~2 min)
  5. account_info.wav       — Информация по счёту (~1 min)
"""

from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

AUDIO_DIR = Path(__file__).parent.parent / "test_data"

VOICE_OPERATOR = "ru-RU-SvetlanaNeural"  # Female operator
VOICE_CLIENT = "ru-RU-DmitryNeural"       # Male client

DIALOGUES: list[dict] = [
    {
        "name": "credit_loan",
        "description": "Потребительский кредит наличными",
        "lines": [
            ("operator", "Добрый день! МТБанк, меня зовут Анна. Чем могу вам помочь?"),
            ("client", "Здравствуйте. Я хотел бы узнать подробнее о кредите наличными."),
            ("operator", "Конечно, с удовольствием расскажу. Какая сумма вас интересует?"),
            ("client", "Хотелось бы получить около десяти тысяч рублей на один год."),
            ("operator", "Хорошо. На данный момент ставка от четырнадцати и девяти процентов годовых. Ежемесячный платёж составит примерно девятьсот рублей."),
            ("client", "А есть ли штраф за досрочное погашение?"),
            ("operator", "Нет, штрафов за досрочное погашение у нас нет. Вы можете погасить кредит в любое время без дополнительных комиссий."),
            ("client", "Отлично. А страховка обязательна?"),
            ("operator", "Страхование жизни — по желанию клиента. Оно не является обязательным условием для получения кредита, однако при наличии страховки ставка может быть немного ниже."),
            ("client", "Понятно. Как можно подать заявку?"),
            ("operator", "Вы можете подать заявку онлайн через наше мобильное приложение МТБанк или прийти в любое отделение. Хотите, я отправлю вам инструкцию по оформлению на электронную почту?"),
            ("client", "Да, пожалуйста. Мой адрес: клиент собака мтбанк точка бай."),
            ("operator", "Записала. Отправлю инструкцию в течение нескольких минут. Есть ли у вас ещё вопросы?"),
            ("client", "Нет, всё понятно. Спасибо большое."),
            ("operator", "Пожалуйста! Если возникнут вопросы — звоните. Всего доброго и хорошего дня!"),
            ("client", "Спасибо, до свидания."),
        ],
    },
    {
        "name": "card_issue",
        "description": "Проблема с банковской картой (8kHz телефония)",
        "telephony": True,
        "lines": [
            ("operator", "Добрый день, МТБанк, оператор Сергей. Слушаю вас."),
            ("client", "Здравствуйте. У меня заблокировалась карта, не могу снять наличные."),
            ("operator", "Понимаю вашу ситуацию. Уточните, пожалуйста, номер вашей карты, последние четыре цифры."),
            ("client", "Да, конечно. Последние цифры: три, семь, два, один."),
            ("operator", "Спасибо. Сейчас проверю. Карта была заблокирована системой безопасности из-за подозрительной транзакции. Для разблокировки нужно подтвердить личность."),
            ("client", "Хорошо, что нужно сделать?"),
            ("operator", "Вам придёт СМС с кодом подтверждения на номер, привязанный к карте. Введите его, пожалуйста."),
            ("client", "Получил. Код: один, два, три, четыре, пять."),
            ("operator", "Отлично, карта разблокирована. Проверьте, пожалуйста, доступность."),
            ("client", "Да, всё работает. Спасибо!"),
            ("operator", "Рад помочь. Если возникнут трудности — обращайтесь. До свидания!"),
        ],
    },
    {
        "name": "complaint",
        "description": "Жалоба на некачественное обслуживание",
        "lines": [
            ("operator", "Добрый день, МТБанк, меня зовут Елена. Чем могу помочь?"),
            ("client", "Здравствуйте. Я хочу подать жалобу на ваш банк! Это возмутительно!"),
            ("operator", "Я вас понимаю и приношу свои извинения. Расскажите, пожалуйста, что произошло?"),
            ("client", "Три дня назад я подал заявку на кредит, мне обещали ответ в течение суток, но до сих пор ничего нет! Это неприемлемо!"),
            ("operator", "Приношу искренние извинения за задержку. Сейчас же проверю статус вашей заявки. Уточните ваше имя и дату рождения для идентификации."),
            ("client", "Михаил Иванов, дата рождения — пятнадцатое марта тысяча девятьсот восемьдесят пятого года."),
            ("operator", "Спасибо. Нашла вашу заявку. Вижу, что она находится на дополнительной проверке. Приношу ещё раз извинения за задержку. Передам ваше обращение в приоритетную очередь и вам перезвонят в течение двух часов."),
            ("client", "Ладно, надеюсь так и будет. Это уже второй раз такая ситуация."),
            ("operator", "Понимаю ваше недовольство. Фиксирую это как повторное обращение. Вам перезвонит старший специалист. Есть ли ещё вопросы?"),
            ("client", "Нет. Жду звонка."),
            ("operator", "Обязательно перезвонят. До свидания!"),
        ],
    },
    {
        "name": "mortgage",
        "description": "Консультация по ипотеке",
        "lines": [
            ("operator", "Добрый день! МТБанк, консультант Ольга. Чем могу помочь?"),
            ("client", "Здравствуйте. Интересует ипотека. Хочу купить квартиру."),
            ("operator", "Замечательно! Расскажите подробнее: какую сумму планируете, на какой срок?"),
            ("client", "Квартира стоит около восьмидесяти тысяч рублей. Первоначальный взнос — двадцать тысяч. Хотелось бы лет на двадцать."),
            ("operator", "Отлично. Сумма кредита составит шестьдесят тысяч рублей. Примерная ставка по ипотеке — от одиннадцати и девяти процентов. Ежемесячный платёж — около шестисот рублей."),
            ("client", "А есть ли программы государственной поддержки?"),
            ("operator", "Да, для семей с детьми действует льготная программа. Если у вас есть дети, ставка может быть снижена. Также есть программа для молодых семей."),
            ("client", "У нас двое детей."),
            ("operator", "Тогда вы можете претендовать на льготную ставку. Рекомендую записаться на консультацию к нашему ипотечному специалисту — он подберёт оптимальный вариант."),
            ("client", "Да, хотелось бы. Как это сделать?"),
            ("operator", "Могу записать вас прямо сейчас. Когда вам удобно: завтра или послезавтра?"),
            ("client", "Лучше завтра, в первой половине дня."),
            ("operator", "Записываю вас на завтра, на одиннадцать утра. Адрес: главный офис, улица Ленина, дом один. Вам придёт СМС-подтверждение."),
            ("client", "Спасибо большое!"),
            ("operator", "Пожалуйста! Всего доброго и до встречи!"),
        ],
    },
    {
        "name": "account_info",
        "description": "Информация по счёту и балансу",
        "lines": [
            ("operator", "Добрый день, МТБанк, оператор Дмитрий. Слушаю вас."),
            ("client", "Здравствуйте. Хочу узнать баланс по карте и посмотреть последние операции."),
            ("operator", "Конечно. Назовите, пожалуйста, последние четыре цифры карты и кодовое слово."),
            ("client", "Карта оканчивается на восемь, девять, три, два. Кодовое слово: весна."),
            ("operator", "Подождите секунду, проверяю. Текущий баланс по вашей карте составляет три тысячи четыреста пятьдесят рублей. Последние операции: вчера оплата в супермаркете — семьдесят рублей, пополнение счёта — пятьсот рублей."),
            ("client", "Хорошо, спасибо. А как подключить уведомления по СМС?"),
            ("operator", "Уведомления можно подключить в мобильном приложении МТБанк в разделе «Настройки» — «Уведомления». Или я могу подключить прямо сейчас."),
            ("client", "Подключите, пожалуйста."),
            ("operator", "Готово. Теперь вы будете получать СМС о каждой транзакции по карте. Ещё что-то могу помочь?"),
            ("client", "Нет, всё. Спасибо!"),
            ("operator", "Пожалуйста. Хорошего дня!"),
        ],
    },
]


async def synthesize_line(text: str, voice: str, output_path: str) -> None:
    """Synthesize a single line using edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def generate_dialogue(dialogue: dict) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    name = dialogue["name"]
    lines = dialogue["lines"]
    telephony = dialogue.get("telephony", False)

    print(f"\n[{name}] Generating {len(lines)} lines...")

    tmp_files: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for i, (speaker, text) in enumerate(lines):
            voice = VOICE_OPERATOR if speaker == "operator" else VOICE_CLIENT
            mp3_path = str(tmp_path / f"{i:03d}_{speaker}.mp3")
            await synthesize_line(text, voice, mp3_path)
            tmp_files.append(mp3_path)
            print(f"  [{speaker}] {text[:50]}...")

        # Create silence file (0.7s)
        silence_path = str(tmp_path / "silence.mp3")
        subprocess.run(
            ["ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", "0.7", "-q:a", "9", "-acodec", "libmp3lame", silence_path, "-y"],
            capture_output=True, check=True
        )

        # Build concat list with silence between lines
        concat_list = str(tmp_path / "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for j, mp3 in enumerate(tmp_files):
                f.write(f"file '{mp3}'\n")
                if j < len(tmp_files) - 1:
                    f.write(f"file '{silence_path}'\n")

        # Concat all
        merged_path = str(tmp_path / "merged.wav")
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", merged_path, "-y"],
            capture_output=True, check=True
        )

        # Final output
        out_path = AUDIO_DIR / f"{name}.wav"

        if telephony:
            # Resample to 8kHz for telephony simulation
            subprocess.run(
                ["ffmpeg", "-i", merged_path,
                 "-ar", "8000", "-ac", "1", str(out_path), "-y"],
                capture_output=True, check=True
            )
        else:
            import shutil
            shutil.copy(merged_path, str(out_path))

    print(f"  [OK] Saved: {out_path}")


async def main() -> None:
    try:
        import edge_tts
    except ImportError:
        print("edge-tts not installed. Run: pip install edge-tts")
        return

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ffmpeg not found. Install it: https://ffmpeg.org/download.html")
        return

    print("Generating test audio files...")
    for dialogue in DIALOGUES:
        await generate_dialogue(dialogue)

    print(f"\nDone! {len(DIALOGUES)} audio files saved to {AUDIO_DIR}/")
    print("\nFiles:")
    for f in sorted(AUDIO_DIR.glob("*.wav")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:30s} {size_kb:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
