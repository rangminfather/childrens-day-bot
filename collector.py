import os
import re
import ssl
import json
import time
import smtplib
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any, Optional

import requests
import feedparser
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KST = timezone(timedelta(hours=9))
NOW_KST = datetime.now(KST)

KEYWORDS = [
    "어린이날 행사 프로그램",
    "어린이날 체험부스",
    "어린이날 프로그램",
    "어린이날 체험프로그램",
    "어린이날 스탬프투어",
    "교회 가족초청 행사",
    "도서관 어린이날 프로그램",
]

SITE_QUERIES = [
    "site:go.kr 어린이날 행사 프로그램",
    "site:or.kr 어린이날 어린이 체험 프로그램",
    "site:museum.go.kr 어린이날 행사",
    "site:library.kr 어린이날 프로그램",
    "site:church 어린이날 행사",
]

INCLUDE_HINTS = [
    "행사", "체험", "프로그램", "축제", "부스", "스탬프", "가족", "초청", "도서관", "박물관", "공연", "포토존", "미션",
]

EXCLUDE_HINTS = [
    "선물", "할인", "상품", "쇼핑", "호텔", "숙박", "펜션", "티켓", "예약판매", "쿠폰",
]

TARGET_SITES = [
    "https://www.nfm.go.kr",
    "https://culture.seoul.go.kr",
    "https://www.sisul.or.kr",
]

SERPER_URL = "https://google.serper.dev/search"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class ConfigError(Exception):
    pass


def env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(f"Missing required env var: {name}")
    return value or ""


def google_news_rss_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=ko&gl=KR&ceid=KR:ko"


def parse_google_news_rss(query: str) -> List[Dict[str, Any]]:
    url = google_news_rss_url(query)
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries:
        link = getattr(e, "link", "")
        title = getattr(e, "title", "").strip()
        published = getattr(e, "published", "")
        summary = BeautifulSoup(getattr(e, "summary", ""), "html.parser").get_text(" ", strip=True)
        items.append({
            "source_type": "rss",
            "query": query,
            "title": title,
            "url": link,
            "snippet": summary,
            "published": published,
        })
    logging.info("RSS %s -> %s items", query, len(items))
    return items


def serper_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    api_key = env("SERPER_API_KEY", required=True)
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "gl": "kr", "hl": "ko", "num": num}
    r = requests.post(SERPER_URL, headers=headers, data=json.dumps(payload), timeout=30)
    r.raise_for_status()
    data = r.json()
    items = []
    for entry in data.get("organic", []):
        items.append({
            "source_type": "search",
            "query": query,
            "title": entry.get("title", "").strip(),
            "url": entry.get("link", ""),
            "snippet": entry.get("snippet", "").strip(),
            "published": entry.get("date", ""),
        })
    logging.info("SERPER %s -> %s items", query, len(items))
    return items


def fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ChildrensDayBot/1.0)"
        })
        r.raise_for_status()
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:4000]
    except Exception as e:
        logging.warning("Failed to fetch %s: %s", url, e)
        return ""


def score_item(item: Dict[str, Any]) -> int:
    text = " ".join([item.get("title", ""), item.get("snippet", ""), item.get("page_text", "")]).lower()
    score = 0

    for kw in INCLUDE_HINTS:
        if kw.lower() in text:
            score += 3
    for kw in EXCLUDE_HINTS:
        if kw.lower() in text:
            score -= 6

    # Prefer government/cultural/library sites over random SEO pages.
    url = item.get("url", "")
    if any(domain in url for domain in ["go.kr", "or.kr", "museum", "library", "church", "ac.kr"]):
        score += 4

    # Prefer pages that clearly talk about actual operation/programs.
    if any(x in text for x in ["프로그램", "체험부스", "참여", "운영", "일정", "공연", "포토존", "미션"]):
        score += 4

    # Prefer seasonal relevance.
    if any(x in text for x in ["어린이날", "5월", "가정의 달"]):
        score += 3

    return score


def normalize_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r"#.*$", "", url)
    return url


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped = []
    seen_urls = set()
    title_keys = []

    for item in items:
        url = normalize_url(item.get("url", ""))
        if not url or url in seen_urls:
            continue

        title = item.get("title", "")
        too_similar = False
        for prev in title_keys:
            if fuzz.token_set_ratio(title, prev) >= 88:
                too_similar = True
                break
        if too_similar:
            continue

        seen_urls.add(url)
        title_keys.append(title)
        item["url"] = url
        deduped.append(item)

    return deduped


def maybe_summarize_with_openai(title: str, snippet: str, page_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    source = (page_text or snippet or title)[:2500]
    if not api_key:
        return heuristic_summary(title, snippet, page_text)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        "다음 자료를 바탕으로 한국어로 1~2문장 요약을 작성하세요. "
        "어린이날 행사 아이디어 관점에서 실제 참고할 만한 요소가 드러나게 쓰세요. "
        "광고 문구처럼 쓰지 말고 담백하게 쓰세요."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "당신은 행사 사례를 짧고 명확하게 요약하는 편집자입니다."},
            {"role": "user", "content": f"제목: {title}\n\n자료:\n{source}\n\n{prompt}"}
        ],
        "temperature": 0.2,
    }
    try:
        r = requests.post(OPENAI_URL, headers=headers, data=json.dumps(payload), timeout=40)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.warning("OpenAI summary failed: %s", e)
        return heuristic_summary(title, snippet, page_text)


def heuristic_summary(title: str, snippet: str, page_text: str) -> str:
    text = page_text or snippet or title
    text = re.sub(r"\s+", " ", text).strip()
    # Pull sentences containing useful cue words first.
    parts = re.split(r"(?<=[.!?。])\s+", text)
    selected = []
    for p in parts:
        if any(k in p for k in ["체험", "프로그램", "공연", "포토존", "미션", "가족", "스탬프", "운영", "부스"]):
            selected.append(p)
        if len(" ".join(selected)) > 180:
            break
    summary = " ".join(selected).strip()
    if not summary:
        summary = text[:180]
    return summary[:220].rstrip()


def collect() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for kw in KEYWORDS:
        items.extend(parse_google_news_rss(kw))
        time.sleep(1)

    search_queries = KEYWORDS + SITE_QUERIES
    for q in search_queries:
        items.extend(serper_search(q, num=8))
        time.sleep(1)

    # Targeted site homepages - low volume but sometimes useful.
    for site in TARGET_SITES:
        items.extend(serper_search(f"site:{site.replace('https://', '').replace('http://', '')} 어린이날 행사", num=5))
        time.sleep(1)

    # Fetch page text selectively.
    for item in items:
        item["page_text"] = fetch_page_text(item["url"])
        item["score"] = score_item(item)

    items = [x for x in items if x["score"] >= 6]
    items.sort(key=lambda x: x["score"], reverse=True)
    items = dedupe_items(items)

    # Keep a manageable pool before summarizing.
    return items[:20]


def format_email(items: List[Dict[str, Any]]) -> str:
    lines = ["[어린이날 행사 아이디어 브리핑]", ""]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item['title']}")
        lines.append(f"   {item['summary']}")
        lines.append(f"   {item['url']}")
        lines.append("")
    lines.append(f"생성 시각(KST): {NOW_KST.strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


def send_email(subject: str, body: str) -> None:
    smtp_host = env("SMTP_HOST", "smtp.naver.com")
    smtp_port = int(env("SMTP_PORT", "587"))
    smtp_user = env("SMTP_USER", required=True)
    smtp_password = env("SMTP_PASSWORD", required=True)
    mail_from = env("MAIL_FROM", smtp_user)
    mail_to = env("MAIL_TO", required=True)

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(smtp_user, smtp_password)
        server.sendmail(mail_from, [mail_to], msg.as_string())


def main() -> None:
    items = collect()
    if not items:
        raise RuntimeError("No items collected. Check filters or API key.")

    final_items = []
    for item in items:
        summary = maybe_summarize_with_openai(item["title"], item.get("snippet", ""), item.get("page_text", ""))
        item["summary"] = summary
        final_items.append(item)
        if len(final_items) >= 7:
            break

    subject = f"[자동발송] 어린이날 행사 아이디어 브리핑 - {NOW_KST.strftime('%Y-%m-%d')}"
    body = format_email(final_items)
    print(body)
    send_email(subject, body)


if __name__ == "__main__":
    main()
