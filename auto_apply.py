#!/usr/bin/env python3
"""
hh.ru Авто-отклик
Поиск → оценка через AI → отклик с сопроводительным → логирование.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# ════════════════════════════════════════════════════════════════
#  Конфигурация
# ════════════════════════════════════════════════════════════════

load_dotenv()

HH_EMAIL = os.getenv("HH_EMAIL", "")
HH_PASSWORD = os.getenv("HH_PASSWORD", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:11434/v1")
AI_MODEL = os.getenv("AI_MODEL", "qwen2.5:14b")
AI_API_KEY = os.getenv("AI_API_KEY", "ollama")
SEARCH_QUERIES = [q.strip() for q in os.getenv("SEARCH_QUERIES", "").split(";") if q.strip()]
AREA = os.getenv("AREA", "1")
SALARY_FROM = os.getenv("SALARY_FROM", "")
COMPANY_EXCLUDE = [c.strip().lower() for c in os.getenv("COMPANY_EXCLUDE", "").split(";") if c.strip()]
EXPERIENCE_PASSES = [e.strip() for e in os.getenv("EXPERIENCE_PASSES", "noExperience;between1And3").split(";") if e.strip()]
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "200"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.65"))

HEADLESS = "--visible" not in sys.argv
BASE_DIR = Path(__file__).resolve().parent
SESSION_DIR = BASE_DIR / "browser_session"
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

TODAY = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = LOGS_DIR / f"{TODAY}.log"
QUESTIONS_FILE = LOGS_DIR / "questions.txt"

# ════════════════════════════════════════════════════════════════
#  Резюме (встроено)
# ════════════════════════════════════════════════════════════════

RESUME_TEXT = """Сафонов Андрей, 22 года, Москва. НИУ МЭИ (теплофизика), высшее.
Лаборант кафедры 2024-2026:
- Автоматизация Excel отчётности (с 2 часов до 10 минут)
- Дипломная: ML в гидродинамике (XGBoost, LightGBM, Random Forest)
- SQL логирование экспериментов (PostgreSQL)
- Дашборд в Tableau
- EDA (Pandas, NumPy, Matplotlib, SciPy)
Навыки: SQL, Python, Pandas, Tableau, XGBoost, LightGBM, Scikit-learn, Git, Docker, Airflow, PostgreSQL, DBeaver"""

# ════════════════════════════════════════════════════════════════
#  Шаблон сопроводительного
# ════════════════════════════════════════════════════════════════

def load_template() -> str:
    path = BASE_DIR / "templates.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ("Здравствуйте! Меня заинтересовала позиция {title}, особенно {interest}. "
            "Это пересекается с моим опытом — {experience}. "
            "Буду рад обсудить подробности в личных сообщениях.")

COVER_TEMPLATE = load_template()

# ════════════════════════════════════════════════════════════════
#  AI клиент
# ════════════════════════════════════════════════════════════════

ai_client = OpenAI(base_url=AI_BASE_URL, api_key=AI_API_KEY)


def ensure_ollama():
    """Запускает Ollama если используется localhost и она не запущена."""
    if "localhost" not in AI_BASE_URL and "127.0.0.1" not in AI_BASE_URL:
        return
    try:
        requests.get(AI_BASE_URL.replace("/v1", ""), timeout=3)
    except Exception:
        print("🦙 Запускаю Ollama...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            time.sleep(1)
            try:
                requests.get(AI_BASE_URL.replace("/v1", ""), timeout=2)
                print("✅ Ollama запущена")
                return
            except Exception:
                pass
        print("⚠️ Не удалось запустить Ollama")


def ai_evaluate(title: str, description: str) -> dict | None:
    """Оценивает вакансию через AI. Возвращает dict с score, reason, interest, experience."""
    prompt = f"""Ты — помощник по поиску работы. Оцени вакансию для кандидата.

РЕЗЮМЕ КАНДИДАТА:
{RESUME_TEXT}

ВАКАНСИЯ: {title}
ОПИСАНИЕ:
{description}

Верни ТОЛЬКО JSON (без markdown, без ```):
{{"score": 0.0-1.0, "reason": "почему такая оценка, 1 предложение", "interest": "конкретная задача из вакансии в дательном падеже, 3-6 слов", "experience": "что кандидат делал похожего из резюме, 8-15 слов"}}

КРИТИЧЕСКИ ВАЖНО — правила оценки:
Кандидат — выпускник без коммерческого опыта. НЕ СНИЖАЙ оценку за "нет опыта работы", "нет коммерческого опыта", "нет опыта в отрасли". Оценивай ТОЛЬКО совпадение технических навыков и задач.

- 0.8-1.0: стек совпадает (SQL, Python, аналитика данных, Tableau, ML — любая комбинация)
- 0.7-0.8: стек частично совпадает, но основные навыки (SQL/Python) есть
- 0.5-0.7: нужны навыки, которых нет (Spark, Scala, 1С, специфичные системы)
- 0.0-0.5: вакансия вообще не про аналитику данных

Снижай оценку ТОЛЬКО если вакансия требует конкретные технологии, которых у кандидата НЕТ (например Spark, Java, 1С, SAP). Отсутствие "опыта в отрасли" (банки, ритейл, медицина) — НЕ причина снижать оценку.

interest — конкретная задача ИЗ ВАКАНСИИ (не общие слова), в дательном падеже.
experience — что кандидат РЕАЛЬНО ДЕЛАЛ из резюме, 8-15 слов."""

    for attempt in range(3):
        try:
            resp = ai_client.chat.completions.create(
                model=AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
            data.pop("cover_letter", None)

            score = float(data.get("score", 0))
            interest = data.get("interest", "").strip()
            experience = data.get("experience", "").strip()
            reason = data.get("reason", "").strip()

            if interest and experience:
                return {"score": score, "reason": reason, "interest": interest, "experience": experience}
            if attempt < 2:
                print(f"  ⟳ AI не вернул interest/experience, попытка {attempt + 2}/3")
                continue
            return {"score": score, "reason": reason, "interest": interest, "experience": experience}
        except Exception as e:
            print(f"  ⚠️ AI ошибка (попытка {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2)
    return None


def ai_answer_question(question: str) -> str:
    """Генерирует ответ на нетиповой вопрос работодателя через AI."""
    prompt = f"""Ты — кандидат на позицию аналитика данных. Ответь кратко на вопрос работодателя.

РЕЗЮМЕ:
{RESUME_TEXT}

ВОПРОС: {question}

Ответь 1-3 предложениями, от первого лица. Без markdown."""
    try:
        resp = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Готов обсудить на интервью. (ошибка AI: {e})"


# ════════════════════════════════════════════════════════════════
#  Логирование
# ════════════════════════════════════════════════════════════════

def log_vacancy(status, title, company, vacancy_id, url,
                score=None, reason="", interest="", experience="", cover=""):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = (
        f"═══════════════════\n"
        f"[{ts}] {status}\n"
        f"───────────────────\n"
        f"Вакансия: {title}\n"
        f"Компания: {company}\n"
        f"ID: {vacancy_id}\n"
        f"URL: {url}\n"
        f"Оценка: {score if score is not None else '—'}\n"
        f"Причина: {reason}\n"
        f"Interest: {interest}\n"
        f"Experience: {experience}\n"
        f"───────────────────\n"
        f"Сопровод: {cover if cover else '—'}\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"  {status} | {title} @ {company} | score={score}")


def log_question(vacancy_title, question, answer):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {vacancy_title}\nВопрос: {question}\nОтвет: {answer}\n{'─' * 40}\n"
    with open(QUESTIONS_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


# ════════════════════════════════════════════════════════════════
#  Фильтрация
# ════════════════════════════════════════════════════════════════

def is_company_excluded(company):
    c = company.lower()
    return any(excl in c for excl in COMPANY_EXCLUDE)


def has_high_experience(text):
    """Ищет требования опыта > 3 лет в тексте."""
    patterns = [
        r"от\s+(\d+)\s*(?:лет|года)",
        r"(\d+)\+?\s*(?:лет|года)\s*(?:опыт|работы)",
        r"опыт\s+(?:от\s+)?(\d+)\s*(?:лет|года)",
        r"(\d+)\s*-\s*\d+\s*(?:лет|года)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if int(m.group(1)) > 3:
                return True
    return False


def is_bank(company):
    bank_kw = ["банк", "bank", "тинькофф", "тинькоф", "tinkoff", "альфа", "alfa",
                "втб", "vtb", "газпромбанк", "райффайзен", "raiffeisen", "росбанк"]
    c = company.lower()
    return any(kw in c for kw in bank_kw)


# ════════════════════════════════════════════════════════════════
#  Зарплата по score
# ════════════════════════════════════════════════════════════════

def salary_by_score(score):
    if score >= 0.85:
        return "150 000 рублей"
    if score >= 0.75:
        return "140 000 рублей"
    if score >= 0.65:
        return "130 000 рублей"
    return "120 000 рублей"


# ════════════════════════════════════════════════════════════════
#  Ответы на вопросы работодателя
# ════════════════════════════════════════════════════════════════

def get_keyword_answer(question, score):
    q = question.lower()
    if any(kw in q for kw in ["зарплат", "оклад", "ожидани", "доход", "руки", "з-п", "з/п"]):
        return salary_by_score(score)
    if any(kw in q for kw in ["гибрид", "офисн"]):
        return "Да, готов рассматривать гибридный и офисный форматы работы."
    if any(kw in q for kw in ["когда готов", "приступить"]):
        return "Готов приступить сразу."
    if re.search(r"\bгород\b", q):
        return "Москва."
    if any(kw in q for kw in ["telegram", "мессенджер", "tg", "телеграм"]):
        return "Telegram: @K_E_T_E_R"
    if re.search(r"уровень.*sql|sql.*уровень|sql.*из.*10|оцени.*sql", q):
        return "7 из 10. Уверенно пишу SELECT, JOIN, подзапросы, оконные функции, CTE."
    if any(kw in q for kw in ["почему ищ", "зачем ищ", "причина поиск"]):
        return "Закончил НИУ МЭИ, ищу позицию в аналитике данных."
    if any(kw in q for kw in ["чем заинтересовала", "почему хотите", "почему выбрали", "что привлекло"]):
        return "Интересны задачи по аналитике данных, стек совпадает с моим опытом."
    if "аутсорсинг" in q or "аутстафф" in q:
        return "Да, такой формат подходит."
    if re.search(r"python.*опыт|опыт.*python|python.*знан|стек.*python", q):
        return "Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib."
    if re.search(r"ml.*опыт|опыт.*ml|machine.?learn|машинн.*обучен", q):
        return "Около 1.5 лет. В дипломной обучал XGBoost, LightGBM, Random Forest."
    if any(kw in q for kw in ["инструмент", "стек", "технологи"]):
        return "SQL (PostgreSQL, DBeaver), Python, Tableau, Excel с макросами."
    return None


# ════════════════════════════════════════════════════════════════
#  Сопроводительное письмо
# ════════════════════════════════════════════════════════════════

def build_cover_letter(title, interest, experience):
    interest = interest[:100].strip() if interest else ""
    experience = experience[:200].strip() if experience else ""
    if not interest or not experience:
        return ""
    cover = COVER_TEMPLATE.format(title=title, interest=interest, experience=experience)
    if not cover.startswith("Здравствуйте"):
        cover = "Здравствуйте! " + cover
    return cover


# ════════════════════════════════════════════════════════════════
#  Утилиты Playwright
# ════════════════════════════════════════════════════════════════

def safe_click(page, locator, timeout=5000):
    """Кликает элемент если он видим, возвращает True при успехе."""
    try:
        el = locator.first
        el.wait_for(state="visible", timeout=timeout)
        el.click()
        return True
    except Exception:
        return False


def wait_and_fill(page, locator, text, timeout=10000):
    """Ждёт появления элемента и вводит текст."""
    el = locator.first
    el.wait_for(state="visible", timeout=timeout)
    el.fill(text)
    return True


def dismiss_popups(page):
    """Закрывает cookies-баннеры и прочие попапы."""
    for text in ["Понятно", "Принять", "OK", "Закрыть"]:
        try:
            btn = page.get_by_role("button", name=text)
            if btn.count() > 0 and btn.first.is_visible(timeout=1000):
                btn.first.click(timeout=2000)
                time.sleep(0.5)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
#  Вход на hh.ru
# ════════════════════════════════════════════════════════════════

def check_logged_in(page):
    """Проверяет, авторизован ли пользователь."""
    try:
        page.goto("https://hh.ru", wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)
        dismiss_popups(page)
        # Ищем элементы, видимые только авторизованным
        for selector in [
            '[data-qa="mainmenu_myResumes"]',
            '[data-qa="mainmenu_responses"]',
            'a[href*="/applicant/resumes"]',
            'a[href*="/applicant/negotiations"]',
        ]:
            if page.locator(selector).count() > 0:
                return True
        # Проверяем наличие аватара/иконки пользователя
        if page.locator('[data-qa="mainmenu_applicantProfile"]').count() > 0:
            return True
    except Exception:
        pass
    return False


def login(page):
    """Логинится на hh.ru. При неудаче в --visible режиме предлагает ручной вход."""
    if check_logged_in(page):
        print("✅ Сессия активна, логин не нужен")
        return True

    print("🔐 Вхожу в аккаунт...")

    try:
        # Шаг 1: Открываем страницу входа
        page.goto("https://hh.ru/account/login", wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)
        dismiss_popups(page)

        # Шаг 2: "Я ищу работу" — может быть кнопкой или ссылкой
        for text_variant in ["Я ищу работу", "Соискатель"]:
            clicked = safe_click(page, page.get_by_text(text_variant, exact=False), timeout=3000)
            if clicked:
                time.sleep(2)
                break

        # Шаг 3: "Войти" — основная кнопка
        safe_click(page, page.get_by_role("button", name="Войти"), timeout=3000)
        time.sleep(1)

        # Шаг 4: Таб "Почта" / "По почте" / "Email"
        for tab_text in ["Почта", "По почте", "Email", "E-mail"]:
            clicked = safe_click(page, page.get_by_text(tab_text, exact=False), timeout=2000)
            if clicked:
                time.sleep(1)
                break

        # Шаг 5: Ввод email — ищем поле ввода несколькими способами
        email_filled = False
        # Способ A: по placeholder
        for placeholder in ["Электронная почта", "Email", "E-mail", "Почта", "Введите email"]:
            try:
                field = page.get_by_placeholder(placeholder)
                if field.count() > 0 and field.first.is_visible(timeout=2000):
                    field.first.fill(HH_EMAIL)
                    email_filled = True
                    break
            except Exception:
                continue
        # Способ B: по типу input
        if not email_filled:
            for selector in [
                'input[type="email"]',
                'input[name="login"]',
                'input[autocomplete="email"]',
                'input[autocomplete="username"]',
                'input[data-qa="account-signup-email"]',
                'input[data-qa="login-input-username"]',
            ]:
                try:
                    field = page.locator(selector)
                    if field.count() > 0 and field.first.is_visible(timeout=2000):
                        field.first.fill(HH_EMAIL)
                        email_filled = True
                        break
                except Exception:
                    continue
        # Способ C: первый видимый текстовый input на странице
        if not email_filled:
            try:
                inputs = page.locator('input[type="text"], input[type="email"], input:not([type])')
                for i in range(inputs.count()):
                    inp = inputs.nth(i)
                    if inp.is_visible(timeout=1000):
                        inp.fill(HH_EMAIL)
                        email_filled = True
                        break
            except Exception:
                pass

        if not email_filled:
            raise Exception("Не нашёл поле для email")

        print("  📧 Email введён")
        time.sleep(1)

        # Шаг 6: "Войти с паролем" — может появиться после ввода email
        for link_text in ["Войти с паролем", "Ввести пароль", "По паролю"]:
            clicked = safe_click(page, page.get_by_text(link_text, exact=False), timeout=3000)
            if clicked:
                time.sleep(1)
                break

        # Шаг 7: Ввод пароля
        pwd_filled = False
        # Ищем поле пароля
        try:
            pwd = page.locator('input[type="password"]')
            pwd.first.wait_for(state="visible", timeout=10000)
            pwd.first.fill(HH_PASSWORD)
            pwd_filled = True
        except Exception:
            pass

        if not pwd_filled:
            for placeholder in ["Пароль", "Password"]:
                try:
                    field = page.get_by_placeholder(placeholder)
                    if field.count() > 0 and field.first.is_visible(timeout=3000):
                        field.first.fill(HH_PASSWORD)
                        pwd_filled = True
                        break
                except Exception:
                    continue

        if not pwd_filled:
            raise Exception("Не нашёл поле для пароля")

        print("  🔑 Пароль введён")
        time.sleep(0.5)

        # Шаг 8: Submit — ищем кнопку отправки
        submitted = False
        for btn_name in ["Войти", "Продолжить", "Далее"]:
            try:
                btn = page.get_by_role("button", name=btn_name)
                if btn.count() > 0:
                    # Берём последнюю видимую кнопку (часто submit внизу)
                    for idx in range(btn.count()):
                        b = btn.nth(idx)
                        if b.is_visible(timeout=1000):
                            b.click()
                            submitted = True
                            break
                if submitted:
                    break
            except Exception:
                continue

        if not submitted:
            # Fallback: submit через Enter
            page.locator('input[type="password"]').first.press("Enter")
            submitted = True

        print("  ⏳ Жду авторизацию...")
        time.sleep(5)

        # Шаг 9: Закрываем "Понятно"
        dismiss_popups(page)

        # Шаг 10: Проверяем успех
        if check_logged_in(page):
            print("✅ Вход выполнен")
            return True

        # Если не залогинились — может быть капча или СМС
        raise Exception("Авторизация не прошла после submit (возможно капча/SMS)")

    except Exception as e:
        print(f"  ⚠️ Автоматический вход не удался: {e}")

        if not HEADLESS:
            print()
            print("╔══════════════════════════════════════════════╗")
            print("║  Войдите вручную в открытом браузере,        ║")
            print("║  затем нажмите Enter здесь.                  ║")
            print("╚══════════════════════════════════════════════╝")
            input()
            if check_logged_in(page):
                print("✅ Ручной вход выполнен, сессия сохранена")
                return True
            print("❌ Не удалось подтвердить вход")
            return False
        else:
            print("💡 Запусти с --visible для ручного входа: ./start.sh --visible")
            return False


# ════════════════════════════════════════════════════════════════
#  Поиск вакансий
# ════════════════════════════════════════════════════════════════

def build_search_url(query, experience):
    params = {"text": query, "area": AREA, "experience": experience, "search_field": "name"}
    if SALARY_FROM:
        params["salary"] = SALARY_FROM
    return "https://hh.ru/search/vacancy?" + urllib.parse.urlencode(params)


def parse_vacancies(page):
    """Парсит список вакансий со страницы поиска."""
    vacancies = []
    # Основной селектор для карточек
    cards = page.locator('a[data-qa="serp-item__title"]')
    count = cards.count()

    if count == 0:
        # Альтернативный селектор
        cards = page.locator('[data-qa="serp-item__title"]')
        count = cards.count()

    for i in range(count):
        try:
            card = cards.nth(i)
            title = card.inner_text(timeout=3000).strip()
            href = card.get_attribute("href", timeout=3000)
            if not href:
                continue

            vid_match = re.search(r"/vacancy/(\d+)", href)
            vacancy_id = vid_match.group(1) if vid_match else ""
            clean_url = href.split("?")[0]
            if not clean_url.startswith("http"):
                clean_url = "https://hh.ru" + clean_url

            # Компания — ищем в DOM
            company = ""
            try:
                # Поднимаемся к карточке вакансии и ищем компанию внутри
                company_text = page.evaluate("""(el) => {
                    let card = el.closest('[data-qa="vacancy-serp__vacancy"]')
                              || el.closest('[class*="serp-item"]')
                              || el.closest('[class*="vacancy-card"]');
                    if (!card) return '';
                    let comp = card.querySelector('[data-qa="vacancy-serp__vacancy-employer"]')
                             || card.querySelector('[data-qa="vacancy-serp__vacancy-employer-text"]')
                             || card.querySelector('[class*="company-name"]');
                    return comp ? comp.innerText.trim() : '';
                }""", card.element_handle())
                company = company_text or ""
            except Exception:
                pass

            vacancies.append({"title": title, "url": clean_url, "id": vacancy_id, "company": company})
        except Exception:
            continue

    return vacancies


# Статусы, которые НЕ записываются в лог-файл.
# Повторные отклики — обычная ситуация при перезапуске скрипта,
# засорять ими лог бессмысленно.
SKIP_WITHOUT_LOG = {
    "Уже откликались ранее",
}


def check_vacancy_status(page):
    """Проверяет статус вакансии ДО траты токенов на AI.
    Возвращает причину пропуска (строку) или None если вакансия доступна."""
    checks = [
        (["Вы откликнулись", "Вы уже откликались", "Резюме доставлено",
          "Отклик отправлен", "Ваш отклик"], "Уже откликались ранее"),
        (["Вам отказали", "Отказ работодателя", "Работодатель отказал",
          "получен отказ", "Отказано"], "Работодатель отказал"),
        (["Вакансия в архиве", "Вакансия перестала публиковаться",
          "Эта вакансия в архиве", "больше не актуальна"], "Вакансия в архиве"),
        (["Приглашение", "Вас пригласили"], "Уже есть приглашение"),
        (["исчерпали лимит", "достигнут лимит откликов"], "Достигнут лимит откликов hh.ru"),
    ]
    for texts, reason in checks:
        for t in texts:
            try:
                if page.get_by_text(t, exact=False).count() > 0:
                    return reason
            except Exception:
                pass
    return None


def open_vacancy(page, url):
    """Открывает вакансию один раз и возвращает (статус_пропуска, описание).
    статус_пропуска = None если вакансию можно обрабатывать."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        dismiss_popups(page)
    except Exception as e:
        return f"Не удалось открыть страницу: {type(e).__name__}", ""

    # Проверяем статус ДО извлечения описания и вызова AI
    status = check_vacancy_status(page)
    if status:
        return status, ""

    # Извлекаем описание
    description = ""
    for selector in [
        '[data-qa="vacancy-description"]',
        '.vacancy-description',
        '[class*="vacancy-description"]',
        'div.vacancy-section',
    ]:
        try:
            el = page.locator(selector)
            if el.count() > 0:
                text = el.first.inner_text(timeout=5000).strip()
                if text:
                    description = text
                    break
        except Exception:
            continue

    if not description:
        try:
            description = page.locator("main, .vacancy-body, article").first \
                              .inner_text(timeout=5000).strip()
        except Exception:
            pass

    if not description:
        return "Описание вакансии пустое или не распознано", ""

    return None, description


# ════════════════════════════════════════════════════════════════
#  Анкета работодателя
# ════════════════════════════════════════════════════════════════

# JS для поиска текста вопроса рядом с полем ввода.
# Ищет элемент ВЫШЕ поля (по bounding rect), пропускает служебные подписи.
_QUESTION_JS = """el => {
    const skip = ["Откликнуться", "Резюме для отклика", "Сопроводительное письмо",
                  "Писать тут", "Добавить", "Отправить", "Отклик на вакансию",
                  "Рекомендуем", "Похожие вакансии"];
    const bad = t => skip.some(s => t.includes(s));

    // 1. Предыдущие соседи
    let prev = el.previousElementSibling;
    for (let i = 0; i < 5 && prev; i++) {
        const t = (prev.innerText || "").trim();
        if (t.length > 5 && t.length < 500 && !bad(t)) return t;
        prev = prev.previousElementSibling;
    }
    // 2. Поднимаемся по дереву, ищем текст НАД полем
    let node = el.parentElement;
    for (let i = 0; i < 10 && node; i++) {
        const candidates = node.querySelectorAll("p, span, label, div, h3, h4, strong, b");
        for (const c of candidates) {
            if (c.contains(el) || el.contains(c)) continue;
            const t = (c.innerText || "").trim();
            if (t.length > 5 && t.length < 500 && !bad(t)) {
                const cRect = c.getBoundingClientRect();
                const elRect = el.getBoundingClientRect();
                if (cRect.bottom <= elRect.top + 50) return t;
            }
        }
        node = node.parentElement;
    }
    // 3. Fallback
    return el.placeholder || el.getAttribute("aria-label") || "";
}"""


def _field_is_empty(field):
    """Проверяет, пустое ли поле ввода."""
    try:
        return not (field.input_value(timeout=2000) or "").strip()
    except Exception:
        return True


def _collect_fields(scope):
    """Возвращает видимые текстовые поля внутри scope, кроме поля сопроводительного."""
    result = []
    try:
        fields = scope.locator('textarea, input[type="text"], input[type="number"]').all()
    except Exception:
        return result

    for f in fields:
        try:
            if not f.is_visible(timeout=1500):
                continue
            dq = (f.get_attribute("data-qa") or "").lower()
            nm = (f.get_attribute("name") or "").lower()
            if "letter" in dq or nm == "text":
                continue
            result.append(f)
        except Exception:
            continue
    return result


def ai_analyze_form(form_text, questions, score):
    """Усиленный режим: AI изучает ВСЮ форму целиком и отвечает на каждый вопрос.
    Возвращает список ответов той же длины, что и questions."""
    salary = salary_by_score(score)
    numbered = "\n".join(f"{i + 1}. {q if q else '(текст вопроса не распознан)'}"
                          for i, q in enumerate(questions))

    prompt = f"""Ты — кандидат на позицию аналитика данных. Работодатель задал вопросы в анкете отклика.

РЕЗЮМЕ КАНДИДАТА:
{RESUME_TEXT}

ДОПОЛНИТЕЛЬНО О КАНДИДАТЕ:
- Город: Москва. Готов к офису и гибриду. Готов приступить сразу.
- Telegram: @K_E_T_E_R
- Зарплатные ожидания: {salary}
- Уровень SQL: 7 из 10 (SELECT, JOIN, подзапросы, оконные функции, CTE)
- Опыт ML: около 1.5 лет (XGBoost, LightGBM, Random Forest в дипломной)

ПОЛНЫЙ ТЕКСТ ФОРМЫ (изучи внимательно, здесь могут быть вопросы,
текст которых не удалось привязать к полям):
---
{form_text[:4000]}
---

ПОЛЯ ДЛЯ ЗАПОЛНЕНИЯ (ровно {len(questions)} штук, по порядку сверху вниз):
{numbered}

Верни ТОЛЬКО JSON-массив из {len(questions)} строк — ответ для каждого поля по порядку.
Формат: ["ответ на поле 1", "ответ на поле 2", ...]

Правила:
- Отвечай кратко (1-2 предложения), от первого лица, по-русски.
- Если поле про зарплату — напиши только "{salary}".
- Если поле про город — "Москва."
- Если поле про контакты/мессенджер — "Telegram: @K_E_T_E_R"
- Если текст вопроса не распознан — посмотри полный текст формы выше
  и определи, о чём спрашивают в этом по счёту поле.
- Никогда не оставляй пустую строку — всегда дай осмысленный ответ."""

    try:
        resp = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        answers = json.loads(raw)
        if isinstance(answers, list):
            answers = [str(a).strip() for a in answers]
            # Дополняем/обрезаем до нужной длины
            while len(answers) < len(questions):
                answers.append("Готов обсудить подробности на собеседовании.")
            return answers[:len(questions)]
    except Exception as e:
        print(f"  ⚠️ AI не разобрал форму: {e}")
    return ["Готов обсудить подробности на собеседовании."] * len(questions)


def _answer_questionnaire(scope, page, score, vacancy_title, thorough=False):
    """Заполняет анкету внутри scope (модалка или страница).
    thorough=True (score > 0.7) — несколько проходов, AI изучает форму целиком.
    Возвращает (успех, количество_заполненных)."""
    time.sleep(1)

    # ── ЕДИНСТВЕННАЯ причина пропуска: тест с кнопками выбора ──
    try:
        if scope.locator('input[type="radio"]').count() > 0:
            print("  ⚠️ Тест с кнопками выбора (radio) — пропускаю вакансию")
            log_question(vacancy_title, "[ТЕСТЫ]", "Пропущено — radio на странице")
            return False, 0
    except Exception:
        pass

    try:
        if scope.locator('input[type="checkbox"]:not([data-qa])').count() > 1:
            print("  ⚠️ Тест с чекбоксами — пропускаю вакансию")
            log_question(vacancy_title, "[ТЕСТЫ]", "Пропущено — checkbox на странице")
            return False, 0
    except Exception:
        pass

    max_passes = 3 if thorough else 1
    total_filled = 0

    if thorough:
        print(f"  🔎 Усиленный режим (score>{0.7}): до {max_passes} проходов по форме")

    for attempt in range(max_passes):
        fields = _collect_fields(scope)
        if not fields:
            if attempt == 0:
                return False, 0
            break

        empty = [f for f in fields if _field_is_empty(f)]
        if not empty:
            print(f"  ✅ Все поля заполнены (проход {attempt + 1})")
            break

        print(f"  📋 Проход {attempt + 1}: полей {len(fields)}, пустых {len(empty)}")

        # Извлекаем текст вопросов
        questions = []
        for f in empty:
            q = ""
            try:
                q = page.evaluate(_QUESTION_JS, f.element_handle())
            except Exception:
                pass
            questions.append(q or "")

        # Подбираем ответы по шаблонам
        answers = [get_keyword_answer(q, score) if q else None for q in questions]
        sources = ["шаблон" if a is not None else None for a in answers]

        # Что не покрыли шаблонами — отдаём AI
        missing = [i for i, a in enumerate(answers) if a is None]
        if missing:
            if thorough:
                # Усиленный режим: AI изучает ВСЮ форму целиком
                try:
                    form_text = scope.inner_text(timeout=5000)
                except Exception:
                    form_text = ""
                ai_answers = ai_analyze_form(form_text, questions, score)
                for i in missing:
                    answers[i] = ai_answers[i]
                    sources[i] = "AI-форма"
            else:
                for i in missing:
                    answers[i] = ai_answer_question(questions[i]) if questions[i] \
                        else "Готов обсудить подробности на собеседовании."
                    sources[i] = "AI"

        # Заполняем
        for field, question, answer, src in zip(empty, questions, answers, sources):
            if not answer:
                continue
            try:
                field.scroll_into_view_if_needed(timeout=3000)
                field.click(timeout=3000)
                time.sleep(0.2)
                field.fill(str(answer))
                time.sleep(0.25)
                # Триггерим события, иначе hh.ru не видит ввод
                try:
                    field.press("End")
                except Exception:
                    pass
                total_filled += 1
                q_show = question[:50] if question else "(вопрос не распознан)"
                print(f"  📝 [{src}] {q_show} → {str(answer)[:40]}")
                log_question(vacancy_title, question or "(не распознан)", str(answer))
            except Exception as e:
                print(f"  ⚠️ Не удалось заполнить поле: {e}")

        if not thorough:
            break

        # Проверяем, разблокировалась ли кнопка отправки
        time.sleep(1)
        if not _submit_is_blocked(scope):
            print(f"  ✅ Кнопка отправки разблокирована (проход {attempt + 1})")
            break
        if attempt < max_passes - 1:
            print("  ⟳ Кнопка ещё заблокирована, повторный проход...")

    return True, total_filled


def _submit_is_blocked(scope):
    """True если кнопка отправки существует и заблокирована."""
    for sel in ['[data-qa="vacancy-response-submit-popup"]',
                 '[data-qa="vacancy-response-letter-submit"]',
                 'button[data-qa*="submit"]']:
        try:
            b = scope.locator(sel)
            if b.count() > 0 and b.first.is_visible(timeout=1500):
                return b.first.is_disabled(timeout=1000)
        except Exception:
            pass
    for name in ["Откликнуться", "Отправить"]:
        try:
            b = scope.get_by_role("button", name=name)
            if b.count() > 0 and b.first.is_visible(timeout=1000):
                return b.first.is_disabled(timeout=1000)
        except Exception:
            pass
    return False



# ════════════════════════════════════════════════════════════════
#  Отклик на вакансию
# ════════════════════════════════════════════════════════════════

def _get_modal(page):
    """Возвращает локатор модального окна или None.
    ВАЖНО: все действия отклика выполняются ВНУТРИ модалки,
    иначе скрипт кликает по рекомендованным вакансиям на фоне."""
    for sel in [
        '[data-qa="vacancy-response-popup"]',
        'div[role="dialog"]',
        '.bloko-modal',
        '[class*="magritte-modal"]',
        '[data-qa*="modal"]',
    ]:
        try:
            loc = page.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=1500):
                return loc.first
        except Exception:
            continue
    return None


def _already_applied(page):
    """Проверяет на СТРАНИЦЕ ВАКАНСИИ, что отклик уже был отправлен."""
    for text in ["Вы откликнулись", "Вы уже откликались", "Резюме доставлено",
                  "Отклик отправлен"]:
        try:
            if page.get_by_text(text, exact=False).count() > 0:
                return True
        except Exception:
            pass
    return False


def _verify_applied(page, url):
    """ЕДИНСТВЕННАЯ надёжная проверка: возвращаемся на страницу вакансии
    и смотрим, исчезла ли кнопка «Откликнуться»."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2.5)
        dismiss_popups(page)
    except Exception:
        return False

    # Есть подтверждающий текст → отклик прошёл
    if _already_applied(page):
        return True

    # Кнопка «Откликнуться» всё ещё активна → отклик НЕ прошёл
    for sel in ['[data-qa="vacancy-response-link-top"]',
                 '[data-qa="vacancy-response-link-bottom"]']:
        try:
            el = page.locator(sel)
            if el.count() > 0 and el.first.is_visible(timeout=2000):
                txt = el.first.inner_text(timeout=2000).strip().lower()
                if "откликнуться" in txt:
                    return False
                if "откликнулись" in txt or "отправлен" in txt:
                    return True
        except Exception:
            pass

    # Кнопки нет и подтверждения нет — неопределённо, считаем неудачей
    return False


def _click_submit(scope):
    """Жмёт кнопку отправки ВНУТРИ scope. Возвращает True если кликнул."""
    # По data-qa
    for sel in [
        '[data-qa="vacancy-response-submit-popup"]',
        '[data-qa="vacancy-response-letter-submit"]',
        'button[data-qa*="submit"]',
    ]:
        try:
            sb = scope.locator(sel)
            if sb.count() > 0 and sb.first.is_visible(timeout=2000):
                if sb.first.is_disabled(timeout=1000):
                    print("  ⚠️ Кнопка отправки заблокирована")
                    return False
                sb.first.click()
                return True
        except Exception:
            pass
    # По тексту
    for btn_text in ["Откликнуться", "Отправить"]:
        try:
            btn = scope.get_by_role("button", name=btn_text)
            if btn.count() > 0 and btn.first.is_visible(timeout=1500):
                if btn.first.is_disabled(timeout=1000):
                    print("  ⚠️ Кнопка отправки заблокирована")
                    return False
                btn.first.click()
                return True
        except Exception:
            pass
    return False


def _fill_cover(scope, cover_letter):
    """Заполняет поле сопроводительного внутри scope."""
    if not cover_letter:
        return False
    for sel in [
        'textarea[data-qa="vacancy-response-popup-form-letter-input"]',
        'textarea[name="text"]',
        'textarea',
    ]:
        try:
            ta = scope.locator(sel)
            if ta.count() > 0 and ta.first.is_visible(timeout=2000):
                ta.first.click(timeout=3000)
                time.sleep(0.2)
                ta.first.fill(cover_letter)
                time.sleep(0.3)
                return True
        except Exception:
            continue
    return False


def apply_to_vacancy(page, vacancy, cover_letter, score):
    """Откликается на вакансию.
    Возвращает (успех, подробная_причина).
    успех = True ТОЛЬКО если отклик подтверждён на странице вакансии."""
    url = vacancy["url"]
    title = vacancy["title"]

    try:
        # ── Всегда открываем страницу вакансии заново ──
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        dismiss_popups(page)

        # Статус вакансии (уже откликались / отказ / архив / лимит)
        status = check_vacancy_status(page)
        if status:
            print(f"  ⏭️ {status}")
            return False, status

        # ── Ищем кнопку «Откликнуться» ТОЛЬКО на странице вакансии ──
        respond_btn = None
        for sel in [
            '[data-qa="vacancy-response-link-top"]',
            '[data-qa="vacancy-response-link-bottom"]',
            '[data-qa="vacancy-response-link-advert"]',
        ]:
            try:
                loc = page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible(timeout=2000):
                    respond_btn = loc.first
                    break
            except Exception:
                continue

        if not respond_btn:
            msg = "Кнопка «Откликнуться» отсутствует на странице"
            print(f"  ⏭️ {msg}")
            return False, msg

        try:
            if respond_btn.is_disabled(timeout=1000):
                msg = "Кнопка «Откликнуться» заблокирована работодателем"
                print(f"  ⏭️ {msg}")
                return False, msg
        except Exception:
            pass

        respond_btn.click()
        time.sleep(3)

        # ── Определяем scope: модалка или страница ──
        modal = _get_modal(page)
        scope = modal if modal else page
        if modal:
            print("  🪟 Открылась модалка отклика")

        # ── Сценарий A: анкета с вопросами работодателя ──
        has_questions = False
        for t in ["ответить на несколько вопросов", "Ответьте на вопросы",
                   "Вопросы работодателя", "ответьте на вопрос"]:
            try:
                if scope.get_by_text(t, exact=False).count() > 0:
                    has_questions = True
                    break
            except Exception:
                pass

        if not has_questions and modal:
            try:
                n = modal.locator('textarea, input[type="text"], input[type="number"]').count()
                if n > 1:
                    has_questions = True
            except Exception:
                pass

        if has_questions:
            thorough = score > 0.7
            if thorough:
                print(f"  🤖 Анкета работодателя (score {round(score, 2)} — усиленный режим)")
            else:
                print("  🤖 Анкета работодателя — отвечаю...")

            ok, filled = _answer_questionnaire(scope, page, score, title, thorough=thorough)
            if not ok:
                return False, "Анкета содержит тест с кнопками выбора (radio/checkbox)"

            if filled == 0:
                return False, "Анкета найдена, но не удалось заполнить ни одного поля"

            # Кнопка «Добавить» раскрывает поле сопроводительного
            if cover_letter:
                for t in ["Добавить сопроводительное", "Добавить"]:
                    try:
                        b = scope.get_by_role("button", name=t)
                        if b.count() > 0 and b.first.is_visible(timeout=1500):
                            b.first.click()
                            time.sleep(0.8)
                            break
                    except Exception:
                        pass
                _fill_cover(scope, cover_letter)

            if not _click_submit(scope):
                if thorough:
                    print("  ⟳ Кнопка заблокирована, финальная попытка дозаполнить...")
                    _answer_questionnaire(scope, page, score, title, thorough=True)
                    time.sleep(1)
                    if _click_submit(scope):
                        time.sleep(3)
                        if _verify_applied(page, url):
                            return True, f"Анкета: заполнено {filled} полей"
                        return False, f"Анкета отправлена ({filled} полей), но hh.ru не подтвердил отклик"
                return False, (f"Анкета заполнена ({filled} полей), "
                                "но кнопка отправки осталась заблокированной")

            time.sleep(3)
            if _verify_applied(page, url):
                return True, f"Анкета: заполнено {filled} полей"
            return False, f"Анкета отправлена ({filled} полей), но hh.ru не подтвердил отклик"

        # ── Сценарий B: попап с полем сопроводительного ──
        has_textarea = False
        try:
            has_textarea = scope.locator("textarea").count() > 0 and \
                           scope.locator("textarea").first.is_visible(timeout=2000)
        except Exception:
            pass

        if has_textarea:
            print("  📝 Попап с сопроводительным")
            cover_ok = _fill_cover(scope, cover_letter)

            if not _click_submit(scope):
                return False, "Попап открыт, но кнопка отправки заблокирована или не найдена"

            time.sleep(3)
            if _verify_applied(page, url):
                return True, ("Попап с письмом" if cover_ok else "Попап без письма")
            return False, "Письмо отправлено, но hh.ru не подтвердил отклик"

        # ── Сценарий C: отклик ушёл сразу («Резюме доставлено») ──
        delivered = False
        for t in ["Резюме доставлено", "Отклик отправлен", "Вы откликнулись"]:
            try:
                if page.get_by_text(t, exact=False).count() > 0:
                    delivered = True
                    break
            except Exception:
                pass

        if delivered:
            print("  ✉️ Резюме доставлено, прикладываю письмо")
            letter_sent = False
            if cover_letter:
                attached = False
                for t in ["Приложить сопроводительное письмо", "Приложить сопроводительное",
                           "Приложить"]:
                    try:
                        b = page.get_by_text(t, exact=False)
                        if b.count() > 0 and b.first.is_visible(timeout=2000):
                            b.first.click()
                            time.sleep(1.5)
                            attached = True
                            break
                    except Exception:
                        pass

                if attached:
                    letter_modal = _get_modal(page)
                    letter_scope = letter_modal if letter_modal else page
                    if _fill_cover(letter_scope, cover_letter):
                        if _click_submit(letter_scope):
                            letter_sent = True
                        time.sleep(2)

            if _verify_applied(page, url):
                return True, ("Мгновенный отклик + письмо" if letter_sent
                               else "Мгновенный отклик без письма")
            return False, "Показано «Резюме доставлено», но отклик не подтвердился"

        # ── Ничего не распознали ──
        print("  ⚠️ Неизвестный сценарий после клика «Откликнуться»")
        if _verify_applied(page, url):
            return True, "Отклик прошёл (сценарий не распознан)"
        return False, "После клика не появилось ни анкеты, ни попапа, ни подтверждения"

    except PwTimeout:
        return False, "Таймаут 30с при отклике (hh.ru не ответил)"
    except Exception as e:
        return False, f"Исключение при отклике: {type(e).__name__}: {str(e)[:120]}"


# ════════════════════════════════════════════════════════════════
#  Главный цикл
# ════════════════════════════════════════════════════════════════

def main():
    ensure_ollama()

    applied_count = 0
    skipped_repeat = 0
    seen_ids = set()

    print(f"🚀 Запуск авто-отклика")
    print(f"   Запросы: {len(SEARCH_QUERIES)}, Опыт: {EXPERIENCE_PASSES}")
    print(f"   Лимит: {DAILY_LIMIT}, MIN_SCORE: {MIN_SCORE}")
    print(f"   Режим: {'видимый' if not HEADLESS else 'headless'}")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch_persistent_context(
            user_data_dir=str(SESSION_DIR),
            headless=HEADLESS,
            viewport={"width": 1280, "height": 900},
            locale="ru-RU",
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        page = browser.pages[0] if browser.pages else browser.new_page()

        if not login(page):
            print("❌ Не удалось войти. Завершаю.")
            if not HEADLESS:
                input("Нажмите Enter для выхода...")
            browser.close()
            return

        # Основной цикл: запрос × опыт
        for query in SEARCH_QUERIES:
            if applied_count >= DAILY_LIMIT:
                break

            for exp_pass in EXPERIENCE_PASSES:
                if applied_count >= DAILY_LIMIT:
                    break

                search_url = build_search_url(query, exp_pass)
                print(f"\n🔍 [{query}] опыт={exp_pass}")

                page_num = 0
                while applied_count < DAILY_LIMIT:
                    current_url = search_url if page_num == 0 else f"{search_url}&page={page_num}"
                    try:
                        page.goto(current_url, wait_until="domcontentloaded", timeout=30000)
                        time.sleep(3)
                        dismiss_popups(page)
                    except Exception:
                        break

                    vacancies = parse_vacancies(page)
                    if not vacancies:
                        break

                    print(f"  📄 Страница {page_num + 1}: {len(vacancies)} вакансий")

                    for vac in vacancies:
                        if applied_count >= DAILY_LIMIT:
                            break

                        vid = vac["id"]
                        if vid in seen_ids:
                            continue
                        seen_ids.add(vid)

                        title = vac["title"]
                        company = vac["company"]

                        # Фильтр по компании
                        if is_company_excluded(company):
                            log_vacancy("🚫 ФИЛЬТР", title, company, vid, vac["url"],
                                        reason="Компания в исключениях")
                            continue

                        # Открываем вакансию: проверка статуса + описание.
                        # Статус проверяется ДО AI, чтобы не тратить токены
                        # на вакансии с отказом / уже откликнулись / архив.
                        skip_reason, desc = open_vacancy(page, vac["url"])
                        if skip_reason:
                            if skip_reason in SKIP_WITHOUT_LOG:
                                # Повторный отклик — в лог не пишем
                                skipped_repeat += 1
                                print(f"  ⏭️ {skip_reason} | {title}")
                            else:
                                log_vacancy("⏭️ ПРОПУСК (статус)", title, company, vid,
                                            vac["url"], reason=skip_reason)
                            continue

                        # Фильтр по опыту
                        if has_high_experience(desc):
                            log_vacancy("⏭️ ПРОПУСК (опыт)", title, company, vid, vac["url"],
                                        reason="В описании указан требуемый опыт больше 3 лет")
                            continue

                        # Оценка AI
                        result = ai_evaluate(title, desc[:3000])
                        if not result:
                            log_vacancy("⚠️ ОШИБКА", title, company, vid, vac["url"],
                                        reason="AI не вернул корректный JSON за 3 попытки")
                            continue

                        score = result["score"]

                        # Бонус для банков
                        if is_bank(company):
                            score = min(score + 0.15, 1.0)
                            result["reason"] += " [+0.15 банк]"

                        if score < MIN_SCORE:
                            log_vacancy("⏭️ ПРОПУСК (оценка)", title, company, vid, vac["url"],
                                        score=round(score, 2),
                                        reason=f"{result['reason']} "
                                               f"(оценка {round(score, 2)} < порога {MIN_SCORE})",
                                        interest=result.get("interest", ""),
                                        experience=result.get("experience", ""))
                            continue

                        # Сопроводительное
                        cover = build_cover_letter(title, result.get("interest", ""),
                                                   result.get("experience", ""))
                        if not cover:
                            print("  ℹ️ Письма не будет: AI не дал interest/experience")

                        # Отклик
                        success, apply_reason = apply_to_vacancy(page, vac, cover, score)
                        if success:
                            applied_count += 1
                            log_vacancy("✅ ОТКЛИК", title, company, vid, vac["url"],
                                        score=round(score, 2),
                                        reason=f"{result['reason']} | {apply_reason}",
                                        interest=result.get("interest", ""),
                                        experience=result.get("experience", ""),
                                        cover=cover)
                            print(f"  📊 Откликов за сессию: {applied_count}")
                        else:
                            if apply_reason in SKIP_WITHOUT_LOG:
                                skipped_repeat += 1
                                print(f"  ⏭️ {apply_reason} | {title}")
                            else:
                                log_vacancy("⚠️ НЕ ОТПРАВЛЕНО", title, company, vid,
                                            vac["url"], score=round(score, 2),
                                            reason=apply_reason,
                                            interest=result.get("interest", ""),
                                            experience=result.get("experience", ""),
                                            cover=cover)

                        time.sleep(2)

                    # Пагинация: возвращаемся на страницу поиска и проверяем «дальше»
                    try:
                        page.goto(current_url, wait_until="domcontentloaded", timeout=30000)
                        time.sleep(2)
                        next_btn = page.locator('[data-qa="pager-next"]')
                        if next_btn.count() > 0 and next_btn.first.is_visible(timeout=3000):
                            page_num += 1
                        else:
                            break
                    except Exception:
                        break

        # Итоги
        print(f"\n{'═' * 40}")
        print(f"📊 Итого: {applied_count} откликов за сессию")
        if skipped_repeat:
            print(f"⏭️ Пропущено повторных (не в логе): {skipped_repeat}")
        print(f"📁 Лог: {LOG_FILE}")
        print(f"{'═' * 40}")

        if not HEADLESS:
            input("\nНажмите Enter для выхода...")

        browser.close()


if __name__ == "__main__":
    main()
