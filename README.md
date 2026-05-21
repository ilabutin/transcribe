# transcribe

Расшифровка аудио/видео с диаризацией спикеров через [WhisperX](https://github.com/m-bain/whisperX).

## Требования

- Python 3.10 (другие версии не тестировались)
- [ffmpeg](https://ffmpeg.org/) — для декодирования аудио
- Аккаунт на [HuggingFace](https://huggingface.co/) и токен доступа

## Установка

### 1. ffmpeg

```bash
brew install ffmpeg
```

### 2. Python 3.10

Если используете pyenv:

```bash
pyenv install 3.10
pyenv local 3.10
```

### 3. Виртуальное окружение

Создать venv **в папке проекта** (не перемещать после создания):

```bash
python3.10 -m venv venv_wx
venv_wx/bin/pip install -r requirements.txt
```

### 4. HuggingFace токен

Диаризация использует модели pyannote, доступные только после принятия лицензий:

1. Создайте токен на https://huggingface.co/settings/tokens
2. Примите условия использования моделей:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0
3. Сохраните токен:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
# или добавьте в ~/.zshrc / ~/.bash_profile
```

## Использование

```bash
# Один файл
./transcribe.sh ~/Downloads/podcast.mp3

# С указанием числа спикеров
./transcribe.sh podcast.mp4 --speakers 2

# Несколько файлов / glob
./transcribe.sh ep1.mp3 ep2.mp3 ep3.mp3 --speakers 2
./transcribe.sh ~/podcasts/*.mp3 --speakers 2

# Указать язык явно (по умолчанию — авто-определение)
./transcribe.sh interview.mp3 --lang en --speakers 2

# Указать путь к выходному файлу (только для одного файла)
./transcribe.sh lecture.mp4 --out ~/Desktop/lecture.txt

# Только транскрипция, без диаризации
./transcribe.sh audio.mp3 --no-diarize

# Меньшая модель (быстрее, качество ниже)
./transcribe.sh audio.mp3 --model medium
```

## Выходной формат

```
0.00 12.45 SPEAKER_00 Текст первого спикера.
12.45 30.10 SPEAKER_01 Текст второго спикера.
```

Промежуточный JSON с сырыми сегментами сохраняется рядом с исходным файлом (`*_whisperx.json`).
