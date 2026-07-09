# Деплой живого демо (OpenWebUI + Pipelines + API) с HTTPS

Полный стек — это три контейнера (`docker compose`), поэтому демо разворачивается
на **VM** с автоматическим HTTPS через Caddy. Ниже — путь на **бесплатной Oracle Cloud
Always Free** (24 ГБ RAM), но те же шаги 3–6 работают на любом VPS (Hetzner, DigitalOcean).

---

## 1. Создать VM

### Вариант A — Oracle Cloud «Always Free» (бесплатно)

1. Регистрация: <https://www.oracle.com/cloud/free/> (нужна карта для верификации — списаний нет).
2. **Compute → Instances → Create Instance**:
   - **Image**: Ubuntu 22.04
   - **Shape**: `VM.Standard.A1.Flex` (ARM Ampere) → 2–4 OCPU, 12–24 ГБ RAM (в пределах Always Free)
   - Добавьте/скачайте SSH-ключ
3. **Networking → Security List** инстанса → **Add Ingress Rules**:
   - `0.0.0.0/0` TCP **80**
   - `0.0.0.0/0` TCP **443**
4. Запомните **публичный IP** инстанса.

> ⚠️ Oracle-образы имеют строгий firewall. После SSH выполните:
> ```bash
> sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
> sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
> sudo netfilter-persistent save
> ```

### Вариант B — VPS (Hetzner CX22 / DigitalOcean, ~€4.5/мес, x86)

Создайте Ubuntu 22.04, 4 ГБ RAM, откройте порты 80/443. Дальше — те же шаги 3–6.

---

## 2. Бесплатный домен (для HTTPS-сертификата)

Let's Encrypt выдаёт сертификат только на домен. Бесплатно:

1. <https://www.duckdns.org> → войдите → создайте поддомен, например `mtbank-demo`.
2. В поле **current ip** впишите публичный IP вашей VM → **update**.
3. Ваш домен: `mtbank-demo.duckdns.org`.

---

## 3. Установить Docker

```bash
ssh ubuntu@ВАШ_IP
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker
```

---

## 4. Развернуть стек

```bash
git clone https://github.com/antonsokol1542-beep/MTBank-AI-Engineer-.git
cd MTBank-AI-Engineer-

# .env с ключами
cp .env.example .env
nano .env          # OPENAI_API_KEY=gsk_... (Groq), при желании HF_TOKEN=hf_...

# домен для Caddy
export DEMO_DOMAIN=mtbank-demo.duckdns.org

# поднять весь стек с HTTPS
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Первый билд API + скачивание Whisper medium — ~10–15 мин. Следите за логами:

```bash
docker compose logs -f api        # ждём "ASR model loaded"
docker compose logs -f caddy      # ждём выдачи сертификата
```

---

## 5. Открыть демо

- **`https://mtbank-demo.duckdns.org`** → откроется чат OpenWebUI (HTTPS ✅).
- Пайплайн подключён автоматически (OpenWebUI ↔ Pipelines настроены в compose).
  В новом чате выберите модель **MTBank Analytics — Full Call Analysis**.
- Прикрепите аудио (скрепка) → получите полный анализ звонка в чате.

Если модель не появилась: **Admin Panel → Settings → Connections** →
проверьте `http://pipelines:9099` с API-ключом из `.env` (`PIPELINES_API_KEY`).

---

## 6. Проверить и отправить

- Демо-ссылка (HTTPS): `https://mtbank-demo.duckdns.org`
- REST API (опционально, если проброшен порт): `POST /analyze`, Swagger `/docs`
- Отклик на файл до 5 мин — < 60 с (medium на CPU VM).

Письмо на **azubik@mtbank.by**:
- 🔗 Репозиторий: `https://github.com/antonsokol1542-beep/MTBank-AI-Engineer-`
- 🌐 Живое демо: `https://mtbank-demo.duckdns.org`

---

## Обслуживание

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart
docker compose -f docker-compose.yml -f docker-compose.prod.yml down     # остановить
docker compose logs -f openwebui                                          # логи чата
```

### Заметки
- **ARM (Oracle):** образы OpenWebUI/Pipelines мультиарх; `torch`/`ctranslate2` ставят
  aarch64-колёса — сборка проходит, но чуть дольше. При сбое `ctranslate2` на ARM
  используйте `WHISPER_MODEL=small`.
- **Мало RAM (< 4 ГБ):** поставьте `WHISPER_MODEL=small` и уберите `HF_TOKEN`.
- **DuckDNS + Caddy:** порты 80 и 443 должны быть открыты и в Security List, и в iptables.
