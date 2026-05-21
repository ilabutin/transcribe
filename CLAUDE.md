# transcribe

CLI-скрипт для расшифровки аудио/видео с диаризацией спикеров на базе WhisperX.

## Структура

- `transcribe.sh` — точка входа, парсит аргументы, запускает Python
- `transcribe_wx.py` — основная логика: транскрипция (WhisperX/faster-whisper), alignment, диаризация (pyannote)
- `venv_wx/` — виртуальное окружение Python 3.10, не в git

## Ключевые детали

**Устройства:** CTranslate2 (транскрипция) не поддерживает MPS — работает на CPU. PyTorch-модели (alignment, диаризация) используют MPS на Apple Silicon, что радикально ускоряет диаризацию.

**Язык:** `--lang` намеренно не имеет дефолтного значения. Авто-определение нужно, потому что подкасты содержат смесь русской речи и англоязычных терминов.

**venv:** После создания нельзя перемещать — пути внутри прошиты абсолютные. Воссоздавать через `python3.10 -m venv venv_wx && venv_wx/bin/pip install -r requirements.txt`.

## Типичный запуск

```bash
./transcribe.sh podcast.mp3 --speakers 2
./transcribe.sh *.mp3 --speakers 2
```
