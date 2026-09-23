#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"

# ── Создание venv при первом запуске ──
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Создаю виртуальное окружение..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    playwright install chromium
    echo "✅ Окружение готово"
else
    source "$VENV_DIR/bin/activate"
fi

# ── Передача аргументов скрипту ──
python3 auto_apply.py "$@"
