from __future__ import annotations

import argparse
import io
import os
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import schedule
import yfinance as yf
from dotenv import load_dotenv
try:
    from deep_translator import GoogleTranslator
except Exception:  # pragma: no cover - optional runtime dependency
    GoogleTranslator = None


KST = ZoneInfo("Asia/Seoul")
HEADERS = {"User-Agent": "Mozilla/5.0"}

MARKET_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "VIXY", "TLT", "UUP", "USO", "GLD"]

SECTOR_ETFS = {
    "XLK": "기술/소프트웨어",
    "SMH": "반도체",
    "IGV": "클라우드/소프트웨어",
    "XLC": "커뮤니케이션/플랫폼",
    "XLY": "임의소비재",
    "XLF": "금융",
    "XLV": "헬스케어",
    "XLI": "산업재",
    "XLE": "에너지",
    "XLU": "유틸리티",
    "XLP": "필수소비재",
    "XLRE": "리츠/부동산",
}


@dataclass(frozen=True)
class TechnicalSnapshot:
    ticker: str
    price: float
    change_pct: float
    volume_ratio: float
    ma20: float
    ma50: float
    ma200: float
    above_200d: bool
    near_52w_high: bool
    golden_cross_watch: bool
    signal: str


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    title: str
    source: str
    title_ko: str = ""


def env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name) or default
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    for fetcher in (fetch_history_yahoo_chart, fetch_history_yfinance, fetch_history_stooq):
        try:
            data = fetcher(ticker, period)
        except Exception:
            data = pd.DataFrame()
        if not data.empty:
            return data
    return pd.DataFrame()


def fetch_history_yahoo_chart(ticker: str, period: str = "1y") -> pd.DataFrame:
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={"range": period, "interval": "1d"},
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()

    node = result[0]
    timestamps = node.get("timestamp") or []
    quote_node = (node.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (node.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    close_values = adjclose or quote_node.get("close") or []
    if not timestamps or not close_values:
        return pd.DataFrame()

    data = pd.DataFrame(
        {
            "Open": quote_node.get("open"),
            "High": quote_node.get("high"),
            "Low": quote_node.get("low"),
            "Close": close_values,
            "Volume": quote_node.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    return data.dropna()


def fetch_history_yfinance(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data.dropna()


def fetch_history_stooq(ticker: str, period: str = "1y") -> pd.DataFrame:
    stooq_symbol = ticker.lower()
    if not stooq_symbol.endswith(".us"):
        stooq_symbol += ".us"

    response = requests.get(
        f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d",
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    if response.text.lstrip().lower().startswith("<!doctype html"):
        return pd.DataFrame()

    data = pd.read_csv(io.StringIO(response.text))
    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date").sort_index()
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=370)
    return data[data.index >= cutoff].dropna()


def to_float(value: object) -> float:
    if isinstance(value, pd.Series):
        return float(value.iloc[0])
    return float(value)


def pct_change(latest: float, previous: float) -> float:
    if previous == 0 or np.isnan(previous):
        return 0.0
    return (latest / previous - 1.0) * 100.0


def analyze_ticker(ticker: str) -> TechnicalSnapshot | None:
    data = fetch_history(ticker)
    if len(data) < 60:
        return None

    close = data["Close"]
    volume = data["Volume"]
    price = to_float(close.iloc[-1])
    previous = to_float(close.iloc[-2])
    change = pct_change(price, previous)
    ma20 = to_float(close.rolling(20).mean().iloc[-1])
    ma50 = to_float(close.rolling(50).mean().iloc[-1])
    ma200 = to_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    avg_volume_20 = to_float(volume.rolling(20).mean().iloc[-1])
    volume_ratio = to_float(volume.iloc[-1]) / avg_volume_20 if avg_volume_20 else 0.0
    high_52w = to_float(close.rolling(min(len(close), 252)).max().iloc[-1])

    above_200d = bool(not np.isnan(ma200) and price > ma200)
    near_52w_high = bool(price >= high_52w * 0.98)
    golden_cross_watch = bool(
        len(close) >= 205
        and ma50 > ma200
        and to_float(close.rolling(50).mean().iloc[-5]) <= to_float(close.rolling(200).mean().iloc[-5])
    )

    primary = change >= 2 or volume_ratio >= 1.8 or near_52w_high or golden_cross_watch
    risk = change <= -2 or not above_200d

    if primary and not risk:
        signal = "특징적 강세"
    elif primary and risk:
        signal = "선별 관찰"
    elif risk:
        signal = "리스크 점검"
    elif above_200d:
        signal = "추세 유지"
    else:
        signal = "관망"

    return TechnicalSnapshot(
        ticker=ticker,
        price=price,
        change_pct=change,
        volume_ratio=volume_ratio,
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        above_200d=above_200d,
        near_52w_high=near_52w_high,
        golden_cross_watch=golden_cross_watch,
        signal=signal,
    )


def fetch_yahoo_news(symbols: list[str], limit: int = 12) -> list[NewsItem]:
    if not symbols:
        return []

    query = ",".join(symbols[:20])
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(query)}&region=US&lang=en-US"
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception:
        return []

    items: list[NewsItem] = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or item.findtext("author") or "Yahoo Finance").strip()
        if title:
            items.append(NewsItem(symbol=query, title=title, source=source, title_ko=translate_to_korean(title)))
        if len(items) >= limit:
            break
    return items


def fetch_news_for_symbols(symbols: list[str], per_symbol: int = 2, total_limit: int = 10) -> list[NewsItem]:
    seen: set[str] = set()
    combined: list[NewsItem] = []
    for symbol in symbols:
        for item in fetch_yahoo_news([symbol], limit=per_symbol):
            key = item.title.lower().strip()
            if key and key not in seen:
                combined.append(NewsItem(symbol=symbol, title=item.title, source=item.source, title_ko=item.title_ko))
                seen.add(key)
            if len(combined) >= total_limit:
                return combined
    return combined


def translate_to_korean(text: str) -> str:
    if not text:
        return ""
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source="auto", target="ko").translate(text)
    except Exception:
        return text


def market_overview() -> list[TechnicalSnapshot]:
    return [snap for ticker in MARKET_TICKERS if (snap := analyze_ticker(ticker))]


def sector_overview() -> list[TechnicalSnapshot]:
    return [snap for ticker in SECTOR_ETFS if (snap := analyze_ticker(ticker))]


def block(title: str, rows: list[str]) -> str:
    if not rows:
        rows = ["해당 없음"]
    return title + "\n" + "\n".join(f"- {row}" for row in rows)


def describe_snapshot(s: TechnicalSnapshot) -> str:
    return f"{s.ticker} {s.change_pct:+.2f}% / 거래량 {s.volume_ratio:.1f}x / {s.signal}"


def build_report() -> str:
    watchlist = env_list("WATCHLIST", "NVDA,MSFT,AAPL,AMZN,GOOGL,META,TSLA,AMD,AVGO,QQQ,SPY")
    screen_universe = env_list(
        "SCREEN_UNIVERSE",
        "NVDA,MSFT,AAPL,AMZN,GOOGL,META,TSLA,AMD,AVGO,MU,SMCI,ARM,PLTR,CRWD,NET,ORCL,ADBE,NFLX,COST,JPM,LLY,UNH,XOM,CVX,GE,BA,CAT,DE",
    )
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    market = market_overview()
    sectors = sector_overview()
    stocks = [snap for ticker in watchlist if (snap := analyze_ticker(ticker))]
    screened = screen_actionable_names(screen_universe, stocks)
    market_news = fetch_news_for_symbols(["SPY", "QQQ", "TLT", "UUP", "USO", "GLD"], per_symbol=1, total_limit=8)
    stock_news = fetch_news_for_symbols(watchlist, per_symbol=2, total_limit=10)

    sections = [
        f"AI Research Desk\n{now}\n\n자동 주문이 아닌 의사결정 보조용 리서치입니다.",
        build_thesis_report(market, sectors, stocks, screened, market_news, stock_news),
        build_cycle_map(market, sectors, stocks),
        build_break_points(market, sectors, stocks),
        build_action_plan(screened, stocks),
        build_actionable_screen_section(screened),
        build_translated_news_digest(market_news, stock_news),
        build_risk_control_section(market, sectors, stocks, screened),
    ]
    return "\n\n".join(sections)


def build_thesis_report(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
    screened: list[TechnicalSnapshot],
    market_news: list[NewsItem],
    stock_news: list[NewsItem],
) -> str:
    by_market = {item.ticker: item for item in market}
    by_stock = {item.ticker: item for item in stocks}
    spy = by_market.get("SPY")
    qqq = by_market.get("QQQ")
    smh = next((s for s in sectors if s.ticker == "SMH"), None)
    xlk = next((s for s in sectors if s.ticker == "XLK"), None)
    xle = next((s for s in sectors if s.ticker == "XLE"), None)
    xlv = next((s for s in sectors if s.ticker == "XLV"), None)
    tlt = by_market.get("TLT")
    uup = by_market.get("UUP")
    uso = by_market.get("USO")
    vixy = by_market.get("VIXY")

    ai_names = [by_stock[t] for t in ("NVDA", "AMD", "AVGO", "MSFT", "META", "GOOGL", "AMZN") if t in by_stock]
    ai_pressure = [s for s in ai_names if s.change_pct < 0]
    ai_resilient = [s for s in ai_names if s.change_pct > 0]

    rows = []
    rows.append("오늘 시장을 한 문장으로 정리하면, 지수의 중기 추세는 아직 살아 있지만 시장 내부에서는 AI/반도체 중심의 고밸류 성장주에서 방어적 섹터로 자금이 일부 이동하는 모습입니다.")

    if smh and xlk:
        rows.append(
            f"특히 반도체(SMH {smh.change_pct:+.2f}%)와 기술주(XLK {xlk.change_pct:+.2f}%)가 약했고, "
            "이는 단순한 하루 변동이라기보다 AI CAPEX 사이클에 대한 시장의 피로감이 가격에 반영되는 과정으로 볼 수 있습니다."
        )

    if uso and uso.change_pct > 1:
        rows.append(
            f"유가 프록시인 USO가 {uso.change_pct:+.2f}% 움직이며 인플레이션 재점화 우려를 건드렸습니다. "
            "유가 상승은 금리와 할인율을 통해 장기 성장주의 멀티플을 직접 압박합니다."
        )

    if tlt and uup:
        rows.append(
            f"TLT {tlt.change_pct:+.2f}%, UUP {uup.change_pct:+.2f}% 조합은 채권과 달러가 동시에 주식시장에 부담을 주는 환경인지 확인하게 만듭니다. "
            "이 조합에서는 좋은 기업도 주가가 쉬어갈 수 있습니다."
        )

    if ai_pressure:
        rows.append(
            "AI 관련 관심종목 중 약세가 확인된 종목은 "
            + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in ai_pressure[:5])
            + "입니다. 이는 AI 스토리가 끝났다는 뜻이 아니라, CAPEX 기대가 이미 상당 부분 가격에 반영되어 있어 작은 금리/마진 우려에도 민감하게 반응한다는 뜻입니다."
        )

    if xle or xlv:
        parts = []
        if xle:
            parts.append(f"에너지 {xle.change_pct:+.2f}%")
        if xlv:
            parts.append(f"헬스케어 {xlv.change_pct:+.2f}%")
        rows.append(
            "반대로 " + ", ".join(parts) + " 흐름은 시장이 완전히 무너지는 것이 아니라, 위험을 줄이면서도 갈 곳을 찾는 로테이션 장세에 가깝다는 점을 보여줍니다."
        )

    rows.append(
        "저의 판단은 이렇습니다. AI 인프라 사이클 자체가 아직 끝났다고 보기는 어렵습니다. "
        "다만 지금부터는 'AI라서 오른다'가 아니라, 실제 클라우드 성장률, CAPEX 지속성, 전력/부지 확보, 빅테크의 FCF 방어력이 확인되는 기업만 살아남는 구간으로 넘어가고 있습니다."
    )

    if screened:
        rows.append(
            "이런 환경에서 신규 관심은 지수 추격이 아니라, 시장이 흔들려도 신고가권 또는 강한 상대강도를 유지하는 "
            + ", ".join(s.ticker for s in screened[:4])
            + " 같은 종목 위주로 좁혀야 합니다."
        )

    rows.append(
        "결론적으로 오늘은 공격적으로 베팅하는 날이 아니라, 다음 사이클에서 살아남을 후보와 탈락할 후보를 구분하는 날입니다. "
        "레버리지는 줄이고, 강한 종목은 눌림과 지지 확인 후 분할로 접근하는 것이 유리합니다."
    )

    return block("1. 핵심 투자 논리", rows)


def build_cycle_map(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
) -> str:
    smh = next((s for s in sectors if s.ticker == "SMH"), None)
    igv = next((s for s in sectors if s.ticker == "IGV"), None)
    xlk = next((s for s in sectors if s.ticker == "XLK"), None)
    xlv = next((s for s in sectors if s.ticker == "XLV"), None)
    xle = next((s for s in sectors if s.ticker == "XLE"), None)

    rows = [
        "현재 AI 사이클은 'CAPEX 확대 → 반도체/데이터센터 인프라 이익 증가 → 하이퍼스케일러 마진 압박 → AI 수익화 검증' 단계 중, 2번에서 3번으로 넘어가는 구간으로 해석합니다.",
        "이 구간에서는 반도체 기업의 실적이 좋아도 섹터 전체가 끝까지 올라가지 못할 수 있습니다. 시장은 이미 다음 질문, 즉 '이 CAPEX가 계속 가능한가?'를 묻기 시작하기 때문입니다.",
    ]

    if smh:
        rows.append(f"반도체(SMH {smh.change_pct:+.2f}%)가 약하다면 이는 실적 부진보다 CAPEX 피크아웃 우려, 마진 압박, 차익실현이 섞인 신호일 가능성이 큽니다.")
    if igv or xlk:
        tech_line = []
        if xlk:
            tech_line.append(f"기술주 {xlk.change_pct:+.2f}%")
        if igv:
            tech_line.append(f"소프트웨어 {igv.change_pct:+.2f}%")
        rows.append(" / ".join(tech_line) + " 흐름은 AI 수혜가 반도체에서 소프트웨어 수익화로 자연스럽게 확산되는지 아직 확인이 필요하다는 뜻입니다.")
    if xlv or xle:
        defensive = []
        if xlv:
            defensive.append(f"헬스케어 {xlv.change_pct:+.2f}%")
        if xle:
            defensive.append(f"에너지 {xle.change_pct:+.2f}%")
        rows.append("상대적으로 " + ", ".join(defensive) + "가 강하다면 시장은 단기적으로 성장주보다 현금흐름과 방어성을 선호하고 있습니다.")

    rows.append("따라서 지금은 AI 사이클 종료가 아니라, AI 사이클 안에서 승자와 후발주가 갈리는 선별 국면으로 보는 것이 더 합리적입니다.")
    return block("2. AI/CAPEX 사이클 진단", rows)


def build_break_points(market: list[TechnicalSnapshot], sectors: list[TechnicalSnapshot]) -> str:
    rows = [
        "이 투자 가설이 깨지는 첫 번째 신호는 하이퍼스케일러들의 클라우드 성장률 둔화입니다. AI 투자가 매출 성장으로 연결되지 않으면 CAPEX 정당성이 약해집니다.",
        "두 번째 신호는 빅테크 FCF 훼손과 신용등급 압박입니다. 초우량 신용등급이 유지되는 동안에는 부채 조달로 투자를 이어갈 수 있지만, 신용 스프레드가 벌어지기 시작하면 이야기가 달라집니다.",
        "세 번째 신호는 전력, 부지, 냉각, 송전망 병목입니다. 돈이 있어도 데이터센터를 실제로 지을 수 없다면 반도체 주문의 가시성이 흔들립니다.",
        "네 번째 신호는 좋은 실적 발표에도 주가가 오르지 못하는 현상입니다. 이것은 기대가 이미 가격에 과하게 반영되었거나, 시장이 다음 사이클 둔화를 보기 시작했다는 뜻일 수 있습니다.",
    ]

    by_market = {item.ticker: item for item in market}
    vixy = by_market.get("VIXY")
    tlt = by_market.get("TLT")
    uup = by_market.get("UUP")
    if vixy and tlt and uup:
        rows.append(
            f"오늘의 매크로 경고등은 VIXY {vixy.change_pct:+.2f}%, TLT {tlt.change_pct:+.2f}%, UUP {uup.change_pct:+.2f}%입니다. "
            "변동성 상승, 채권 약세, 달러 강세가 동시에 나타나면 성장주 비중 확대는 보수적으로 봐야 합니다."
        )

    rows.append("반대로 이 가설이 유지되는 조건은 클라우드 성장률 견조, CAPEX 가이던스 유지, AI 서비스 매출화, 전력 인프라 투자 확대입니다.")
    return block("3. 무엇을 보면 판단을 바꿀 것인가", rows)


def build_action_plan(screened: list[TechnicalSnapshot], stocks: list[TechnicalSnapshot]) -> str:
    weak = [s for s in stocks if s.signal == "리스크 점검"]
    rows = [
        "지금은 포지션을 크게 늘리는 구간이 아니라, 강한 종목이 조정 중에도 살아남는지 확인하는 구간입니다.",
        "신규 매수는 한 번에 들어가기보다 1차 관찰, 2차 지지 확인, 3차 거래량 동반 재상승 확인 순서가 좋습니다.",
    ]
    if screened:
        rows.append(
            "우선 관찰 후보는 "
            + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in screened[:5])
            + "입니다. 이 종목들은 시장이 약해도 상대강도가 유지되는지 확인할 가치가 있습니다."
        )
    if weak:
        rows.append(
            "주의 후보는 "
            + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in weak[:5])
            + "입니다. 반등이 나오더라도 거래량 없는 반등이면 비중 확대 근거로 보기 어렵습니다."
        )
    rows.append("레버리지는 특히 조심해야 합니다. 지금 같은 CAPEX 의존 장세는 방향은 맞아도 변동성으로 먼저 흔들릴 수 있습니다.")
    rows.append("벌 때 크게 버는 것보다, 틀렸을 때 계좌가 살아남는 구조를 만드는 것이 다음 사이클 베팅의 전제입니다.")
    return block("4. 오늘의 행동 지침", rows)


def build_translated_news_digest(market_news: list[NewsItem], stock_news: list[NewsItem]) -> str:
    rows = []
    combined = market_news[:4] + stock_news[:6]
    for item in combined[:8]:
        rows.append(f"{item.title_ko} ({item.source})")
    rows.append("뉴스 해석 원칙: 헤드라인은 재료이고, 진짜 판단은 가격 반응입니다. 좋은 뉴스에도 못 오르면 기대 선반영, 나쁜 뉴스에도 버티면 수급 우위로 해석합니다.")
    return block("6. 번역 뉴스와 해석 포인트", rows)


def build_cio_memo(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
    screened: list[TechnicalSnapshot],
) -> str:
    tone = infer_market_tone(market)
    leaders = sorted(sectors, key=lambda s: s.change_pct, reverse=True)[:3]
    laggards = sorted(sectors, key=lambda s: s.change_pct)[:3]
    by_ticker = {item.ticker: item for item in market}
    spy = by_ticker.get("SPY")
    qqq = by_ticker.get("QQQ")
    vixy = by_ticker.get("VIXY")
    tlt = by_ticker.get("TLT")
    uup = by_ticker.get("UUP")

    stance = "중립"
    if spy and qqq and vixy:
        if spy.above_200d and qqq.above_200d and vixy.change_pct < 0:
            stance = "선별적 위험 선호"
        elif vixy.change_pct > 0 and ((tlt and tlt.change_pct < 0) or (uup and uup.change_pct > 0)):
            stance = "방어적 운용"

    rows = [
        f"운용 스탠스: {stance}. {tone}",
        "오늘의 조언: 시장 전체를 추격하기보다, 강한 섹터 안에서 뉴스 촉매와 가격 확인이 동시에 나온 종목만 선별합니다.",
    ]
    if leaders:
        rows.append("주도 섹터: " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)}({s.change_pct:+.1f}%)" for s in leaders))
    if laggards:
        rows.append("부진 섹터: " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)}({s.change_pct:+.1f}%)" for s in laggards))
    if screened:
        rows.append("발굴 후보: " + ", ".join(f"{s.ticker}({s.signal}, {s.change_pct:+.1f}%)" for s in screened[:5]))
    if stocks:
        weak = [s for s in stocks if s.signal == "리스크 점검"]
        if weak:
            rows.append("주의 후보: " + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in weak[:4]))
    rows.append("의사결정 원칙: 매수는 '매크로 부담이 완화되거나 주도 섹터가 뚜렷할 때', 매도/축소는 '좋은 뉴스에도 주가가 반응하지 않을 때' 우선 검토합니다.")
    return block("1. CIO 데스크 의견", rows)


def infer_market_tone(market: list[TechnicalSnapshot]) -> str:
    by_ticker = {item.ticker: item for item in market}
    spy = by_ticker.get("SPY")
    qqq = by_ticker.get("QQQ")
    iwm = by_ticker.get("IWM")
    vixy = by_ticker.get("VIXY")
    tlt = by_ticker.get("TLT")
    uup = by_ticker.get("UUP")

    if not spy or not qqq:
        return "시장 데이터가 부족합니다. 뉴스와 개별 종목 신호 중심으로 해석합니다."

    score = 0
    score += 1 if spy.change_pct > 0 else -1
    score += 1 if qqq.change_pct > spy.change_pct else 0
    score += 1 if iwm and iwm.change_pct > spy.change_pct else -1
    score += 1 if vixy and vixy.change_pct < 0 else -1
    score += 1 if tlt and tlt.change_pct > 0 else -1
    score += 1 if uup and uup.change_pct < 0 else -1

    if score >= 2:
        return "시장 톤은 위험 선호에 가깝습니다. 성장주와 경기민감 섹터가 동반 강세인지 확인합니다."
    if score <= -2:
        return "시장 톤은 방어적입니다. 금리/달러/변동성 부담이 주식 멀티플을 압박하는지 점검합니다."
    return "시장 톤은 혼조입니다. 지수 방향보다 섹터 로테이션과 개별 촉매의 질이 더 중요합니다."


def build_macro_section(market: list[TechnicalSnapshot], news: list[NewsItem]) -> str:
    by_ticker = {item.ticker: item for item in market}

    positives: list[str] = []
    negatives: list[str] = []
    interpretation: list[str] = []

    spy = by_ticker.get("SPY")
    qqq = by_ticker.get("QQQ")
    iwm = by_ticker.get("IWM")
    vixy = by_ticker.get("VIXY")
    tlt = by_ticker.get("TLT")
    uup = by_ticker.get("UUP")
    uso = by_ticker.get("USO")
    gld = by_ticker.get("GLD")

    if spy and spy.above_200d:
        positives.append("SPY가 200일선 위에 있어 중기 상승 추세는 아직 유효합니다.")
    if qqq and spy and qqq.change_pct > spy.change_pct:
        positives.append("QQQ가 SPY를 앞서면 성장주 선호가 유지되는 신호입니다.")
    if iwm and spy and iwm.change_pct > spy.change_pct:
        positives.append("IWM 상대강도 개선은 랠리 폭이 넓어지는 신호입니다.")
    if vixy and vixy.change_pct > 0:
        negatives.append("변동성 상품 상승은 위험 회피 수요가 커지고 있음을 시사합니다.")
    if tlt and tlt.change_pct < 0:
        negatives.append("TLT 약세는 장기금리 상승 압력으로 해석될 수 있어 성장주 밸류에이션에 부담입니다.")
    if uup and uup.change_pct > 0:
        negatives.append("달러 강세는 글로벌 매출 비중이 큰 대형 기술주에 단기 부담이 될 수 있습니다.")
    if uso and abs(uso.change_pct) >= 2:
        direction = "상승" if uso.change_pct > 0 else "하락"
        interpretation.append(f"유가 프록시 USO가 {direction}했습니다. 에너지/운송/소비재 마진 영향까지 함께 봐야 합니다.")
    if gld and gld.change_pct > 1:
        interpretation.append("금 가격 강세는 방어 수요 또는 실질금리 기대 변화를 반영할 수 있습니다.")

    headline_rows = [f"{item.title_ko} ({item.source})" for item in news[:4]]

    rows = []
    rows.extend(["긍정 근거: " + item for item in positives[:3]])
    rows.extend(["부정 근거: " + item for item in negatives[:3]])
    rows.extend(["해석: " + item for item in interpretation[:3]])
    rows.extend(["주요 헤드라인: " + item for item in headline_rows])
    if negatives and positives:
        rows.append("전략 판단: 추세는 살아 있지만 할인율 부담이 커질 수 있는 환경입니다. 강한 종목도 분할 접근이 더 적합합니다.")
    elif negatives:
        rows.append("전략 판단: 신규 매수보다 현금 비중, 손절 기준, 이벤트 리스크를 먼저 점검합니다.")
    elif positives:
        rows.append("전략 판단: 주도주 눌림목과 신고가 돌파 후보를 선별적으로 검토할 수 있습니다.")
    return block("2. 매크로 데스크", rows)


def build_sector_section(sectors: list[TechnicalSnapshot], news: list[NewsItem]) -> str:
    if not sectors:
        return block("3. 산업/섹터 데스크", ["섹터 ETF 데이터를 충분히 수집하지 못했습니다."])

    leaders = sorted(sectors, key=lambda s: s.change_pct, reverse=True)[:4]
    laggards = sorted(sectors, key=lambda s: s.change_pct)[:4]
    rows = []

    rows.append("강세 산업: " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)} {s.change_pct:+.2f}%" for s in leaders))
    rows.append("약세 산업: " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)} {s.change_pct:+.2f}%" for s in laggards))

    for sector in leaders[:3]:
        label = SECTOR_ETFS.get(sector.ticker, sector.ticker)
        if sector.volume_ratio >= 1.4:
            rows.append(f"{label}: 가격 상승과 거래량 증가가 동반되어 기관성 관심 여부를 확인할 가치가 있습니다.")
        elif sector.near_52w_high:
            rows.append(f"{label}: 신고가권 접근. 뉴스가 실적 추정치 상향으로 이어지는지가 핵심입니다.")
        else:
            rows.append(f"{label}: 상대강도는 양호하지만 거래량 확인 전까지는 추격보다 관찰이 유리합니다.")

    sector_headlines = [item.title_ko for item in news[:5]]
    if sector_headlines:
        rows.append("관련 뉴스 테마: " + " / ".join(sector_headlines[:3]))

    return block("3. 산업/섹터 데스크", rows)


def screen_actionable_names(universe: list[str], watchlist_snaps: list[TechnicalSnapshot]) -> list[TechnicalSnapshot]:
    excluded = set(MARKET_TICKERS) | set(SECTOR_ETFS)
    seen = {snap.ticker for snap in watchlist_snaps}
    snaps = list(watchlist_snaps)
    for ticker in universe:
        if ticker not in seen and (snap := analyze_ticker(ticker)):
            snaps.append(snap)
            seen.add(ticker)

    def score(s: TechnicalSnapshot) -> float:
        score_value = 0.0
        score_value += 3.0 if s.near_52w_high else 0.0
        score_value += 2.5 if s.volume_ratio >= 1.8 and s.change_pct > 0 else 0.0
        score_value += 2.0 if s.golden_cross_watch else 0.0
        score_value += 1.5 if s.above_200d else -2.0
        score_value += min(max(s.change_pct, -5), 5) / 2.0
        score_value -= 2.0 if s.change_pct <= -4 else 0.0
        return score_value

    candidates = [
        s
        for s in snaps
        if s.ticker not in excluded
        and (
        (s.near_52w_high and s.above_200d and s.change_pct > 0)
        or (s.volume_ratio >= 1.8 and s.change_pct > 1)
        or s.golden_cross_watch
        or (s.change_pct >= 4 and s.above_200d)
        )
    ]
    return sorted(candidates, key=score, reverse=True)[:10]


def build_actionable_screen_section(screened: list[TechnicalSnapshot]) -> str:
    rows = []
    for s in screened[:8]:
        reason = []
        if s.near_52w_high:
            reason.append("52주 신고가권")
        if s.volume_ratio >= 1.8:
            reason.append(f"거래량 {s.volume_ratio:.1f}배")
        if s.golden_cross_watch:
            reason.append("골든크로스")
        if s.change_pct >= 4:
            reason.append("강한 가격 모멘텀")
        if s.above_200d:
            reason.append("200일선 상회")
        support = f"1차 지지 ${s.ma20:.2f}, 추세 훼손 ${s.ma50:.2f}"
        if not np.isnan(s.ma200):
            support += f", 장기 방어선 ${s.ma200:.2f}"
        playbook = "돌파형 후보" if s.near_52w_high or s.change_pct >= 4 else "수급 확인 후보"
        rows.append(
            f"{s.ticker}: {s.change_pct:+.2f}%, {', '.join(reason)}. "
            f"{playbook}. 확인: 상승일 거래량 유지와 다음 거래일 시초가 방어. "
            f"무효화: {support} 이탈."
        )
    rows.append("스크리닝 기준: 신고가권, 거래량 급증, 골든크로스, 강한 양봉, 200일선 상회가 겹치는 종목을 우선 탐지합니다.")
    return block("5. 액션 가능 종목 스크리닝", rows)


def build_event_section(news: list[NewsItem]) -> str:
    rows = []
    for item in news[:8]:
        rows.append(f"{item.title_ko} ({item.source})")
    rows.append("해석 기준: 좋은 뉴스에도 주가가 하락하면 차익실현/기대 선반영을 의심하고, 나쁜 뉴스에도 버티면 수급 강도를 높게 평가합니다.")
    return block("5. 뉴스/이벤트 레이더", rows)


def build_compact_technical_section(stocks: list[TechnicalSnapshot]) -> str:
    unusual = [
        s
        for s in stocks
        if s.volume_ratio >= 1.8 or s.golden_cross_watch or s.near_52w_high or abs(s.change_pct) >= 4
    ]
    risk = [s for s in stocks if s.signal == "리스크 점검"]

    rows = []
    rows.extend("특징 신호: " + describe_snapshot(s) for s in unusual[:5])
    rows.extend("리스크 후보: " + describe_snapshot(s) for s in risk[:5])
    return block("5. 기술적 레이더", rows)


def build_risk_control_section(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
    screened: list[TechnicalSnapshot],
) -> str:
    by_ticker = {item.ticker: item for item in market}
    spy = by_ticker.get("SPY")
    vixy = by_ticker.get("VIXY")
    tlt = by_ticker.get("TLT")
    uup = by_ticker.get("UUP")

    leaders = sorted(sectors, key=lambda s: s.change_pct, reverse=True)[:2]
    notable = screened[:3]

    rows = []
    rows.append("기본 결론: 신규 매수 판단은 매크로 톤, 주도 섹터, 개별 촉매가 같은 방향일 때만 강화합니다.")

    positive_parts = []
    if spy and spy.above_200d:
        positive_parts.append("지수 중기 추세 유지")
    if leaders:
        positive_parts.append("주도 섹터 존재")
    if notable:
        positive_parts.append("관심종목 내 특징 신호 존재")
    rows.append("매수/비중확대 근거: " + (", ".join(positive_parts) if positive_parts else "아직 뚜렷하지 않음"))

    negative_parts = []
    if vixy and vixy.change_pct > 0:
        negative_parts.append("변동성 상승")
    if tlt and tlt.change_pct < 0:
        negative_parts.append("금리 부담 가능성")
    if uup and uup.change_pct > 0:
        negative_parts.append("달러 강세")
    rows.append("보류/축소 근거: " + (", ".join(negative_parts) if negative_parts else "현재 리스크 신호는 제한적"))

    if notable:
        rows.append("우선 관찰 종목: " + ", ".join(f"{s.ticker}({s.signal}, {s.change_pct:+.1f}%)" for s in notable))
    rows.append("다음 확인 포인트: 뉴스가 실적 전망 변화로 연결되는지, 강세 섹터에 거래량이 붙는지, 지수 하락 시 주도주가 버티는지.")

    rows.append("리스크 관리: 하루 변동성이 큰 종목은 첫 진입 비중을 줄이고, 실적/규제/금리 이벤트 전에는 신규 비중 확대를 보수적으로 처리합니다.")
    return block("7. 투자위원회 리스크 관리", rows)


def classify_market_regime(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
) -> dict[str, object]:
    by_market = {item.ticker: item for item in market}
    by_sector = {item.ticker: item for item in sectors}
    by_stock = {item.ticker: item for item in stocks}

    spy = by_market.get("SPY")
    qqq = by_market.get("QQQ")
    iwm = by_market.get("IWM")
    vixy = by_market.get("VIXY")
    tlt = by_market.get("TLT")
    uup = by_market.get("UUP")
    uso = by_market.get("USO")
    smh = by_sector.get("SMH")
    xlk = by_sector.get("XLK")
    igv = by_sector.get("IGV")
    xle = by_sector.get("XLE")
    xlv = by_sector.get("XLV")
    xlp = by_sector.get("XLP")
    xlu = by_sector.get("XLU")

    ai_core = [by_stock[t] for t in ("NVDA", "AMD", "AVGO", "MSFT", "META", "GOOGL", "AMZN") if t in by_stock]
    ai_avg = sum(s.change_pct for s in ai_core) / len(ai_core) if ai_core else 0.0
    defensive_avg = np.mean([s.change_pct for s in (xlv, xlp, xlu) if s is not None]) if any([xlv, xlp, xlu]) else 0.0

    top_sectors = sorted(sectors, key=lambda s: s.change_pct, reverse=True)[:3]
    weak_sectors = sorted(sectors, key=lambda s: s.change_pct)[:3]

    if smh and xlk and smh.change_pct > 1 and xlk.change_pct > 0 and ai_avg > 0:
        regime = "AI/반도체 반등"
        stance = "선별적 위험 선호"
        thesis = "전일과 달리 오늘은 반도체와 기술주가 시장을 다시 끌어올리는 쪽으로 성격이 바뀌었습니다. 다만 빅테크 전반이 모두 강한 것이 아니라, AI 인프라 체인 안에서도 승자와 후발주가 갈리는 반등입니다."
        action = "강한 AI 인프라 종목은 관찰하되, 전일 약세를 하루 만에 모두 무효화했다고 보기보다 거래량과 종가 위치를 확인해야 합니다."
    elif xle and uso and xle.change_pct > 1 and uso.change_pct > 1:
        regime = "유가/인플레이션 로테이션"
        stance = "방어적 선별"
        thesis = "오늘 시장의 중심은 성장주보다 유가와 인플레이션 프록시입니다. 유가 상승이 에너지에는 호재지만, 장기 성장주에는 할인율 부담으로 작용할 수 있습니다."
        action = "에너지는 추격보다 눌림 확인, 기술주는 금리와 유가가 진정되는지 확인한 뒤 접근하는 것이 적합합니다."
    elif defensive_avg > 0.5 and qqq and spy and qqq.change_pct < spy.change_pct:
        regime = "방어주 로테이션"
        stance = "방어적 운용"
        thesis = "시장 내부에서 성장주보다 헬스케어, 필수소비재, 유틸리티 같은 방어 섹터가 선호되고 있습니다. 이는 지수보다 리스크 관리가 중요한 장세입니다."
        action = "신규 매수는 방어적 성장주와 현금흐름이 안정적인 종목 위주로 제한하는 편이 낫습니다."
    elif iwm and spy and iwm.change_pct > spy.change_pct and vixy and vixy.change_pct <= 0:
        regime = "시장 폭 확대"
        stance = "점진적 위험 선호"
        thesis = "대형 기술주만 오르는 장세가 아니라 중소형주까지 참여하는 폭 확대 흐름이 나타나고 있습니다. 이는 랠리의 질이 좋아지는 신호일 수 있습니다."
        action = "주도주만 추격하기보다, 신고가권에 가까운 2선 종목과 섹터 내 후발 강세주를 함께 봅니다."
    elif spy and spy.above_200d and qqq and qqq.change_pct < 0:
        regime = "상승 추세 내 차익실현"
        stance = "중립/분할 접근"
        thesis = "지수의 중기 추세는 살아 있지만 고밸류 성장주에서 차익실현이 나오고 있습니다. 구조적 하락이라기보다 상승 사이클 중 속도 조절로 해석할 여지가 있습니다."
        action = "강한 종목은 바로 추격하지 말고, 지지선 확인 후 분할로 접근합니다."
    else:
        regime = "혼조/선별 장세"
        stance = "중립"
        thesis = "시장 전체의 방향성보다 섹터와 종목별 차별화가 더 중요한 장세입니다. 같은 AI, 같은 빅테크 안에서도 가격 반응이 달라지고 있습니다."
        action = "뉴스보다 가격 반응을 우선하고, 강한 종목과 약한 종목을 명확히 분리합니다."

    return {
        "regime": regime,
        "stance": stance,
        "thesis": thesis,
        "action": action,
        "top_sectors": top_sectors,
        "weak_sectors": weak_sectors,
        "ai_avg": ai_avg,
        "defensive_avg": defensive_avg,
        "macro": {"spy": spy, "qqq": qqq, "vixy": vixy, "tlt": tlt, "uup": uup, "uso": uso, "smh": smh, "xlk": xlk, "igv": igv, "xle": xle},
    }


def build_thesis_report(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
    screened: list[TechnicalSnapshot],
    market_news: list[NewsItem],
    stock_news: list[NewsItem],
) -> str:
    regime = classify_market_regime(market, sectors, stocks)
    macro = regime["macro"]
    rows = [
        f"오늘의 시장 성격: {regime['regime']}. 운용 스탠스는 {regime['stance']}입니다.",
        str(regime["thesis"]),
    ]

    top = regime["top_sectors"]
    weak = regime["weak_sectors"]
    if top:
        rows.append("오늘 돈이 들어온 곳은 " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)}({s.change_pct:+.2f}%)" for s in top) + "입니다.")
    if weak:
        rows.append("반대로 약했던 곳은 " + ", ".join(f"{SECTOR_ETFS.get(s.ticker, s.ticker)}({s.change_pct:+.2f}%)" for s in weak) + "입니다.")

    smh = macro.get("smh")
    xlk = macro.get("xlk")
    if smh and xlk:
        if smh.change_pct > 0 and xlk.change_pct > 0:
            rows.append(f"반도체(SMH {smh.change_pct:+.2f}%)와 기술주(XLK {xlk.change_pct:+.2f}%)가 동반 상승했기 때문에, 오늘은 적어도 'AI/반도체가 시장을 끌어내린 날'은 아닙니다.")
        elif smh.change_pct < 0 and xlk.change_pct < 0:
            rows.append(f"반도체(SMH {smh.change_pct:+.2f}%)와 기술주(XLK {xlk.change_pct:+.2f}%)가 함께 약해, 성장주 밸류에이션 부담이 시장의 핵심 압력으로 작용했습니다.")
        else:
            rows.append(f"반도체(SMH {smh.change_pct:+.2f}%)와 기술주(XLK {xlk.change_pct:+.2f}%)가 엇갈려, AI 체인 내부에서도 종목 선별이 강해졌습니다.")

    uso = macro.get("uso")
    tlt = macro.get("tlt")
    uup = macro.get("uup")
    vixy = macro.get("vixy")
    if uso and abs(uso.change_pct) >= 1:
        direction = "상승" if uso.change_pct > 0 else "하락"
        rows.append(f"유가 프록시 USO는 {uso.change_pct:+.2f}%로 {direction}했습니다. 이는 에너지, 인플레이션 기대, 장기금리 해석에 직접 연결됩니다.")
    if tlt and uup and vixy:
        rows.append(f"매크로 압력판은 TLT {tlt.change_pct:+.2f}%, UUP {uup.change_pct:+.2f}%, VIXY {vixy.change_pct:+.2f}%입니다. 이 셋이 동시에 주식에 불리하게 움직이는지 여부가 오늘 리스크 판단의 핵심입니다.")

    if screened:
        rows.append("현재 스크리닝상 살아있는 후보는 " + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in screened[:5]) + "입니다.")
    rows.append(str(regime["action"]))
    rows.append("결론적으로, 오늘 보고서는 고정된 AI/CAPEX 서사를 반복하지 않고 실제 섹터 흐름과 가격 반응을 우선합니다.")
    return block("1. 핵심 투자 논리", rows)


def build_cycle_map(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
) -> str:
    regime = classify_market_regime(market, sectors, stocks)
    macro = regime["macro"]
    rows = [f"오늘 적용할 사이클 프레임은 '{regime['regime']}'입니다."]

    if regime["regime"] == "AI/반도체 반등":
        rows.extend([
            "오늘은 AI CAPEX 우려보다 AI 인프라 체인의 반등 탄력이 더 중요합니다. 다만 모든 AI 종목을 같은 바구니로 보면 안 됩니다.",
            "반도체가 강한 날에는 후속 확인 포인트가 분명합니다. 상승이 NVDA/AVGO/SMH 같은 인프라 대장주에 집중되는지, 아니면 소프트웨어와 클라우드까지 확산되는지 봐야 합니다.",
        ])
    elif regime["regime"] == "유가/인플레이션 로테이션":
        rows.extend([
            "오늘은 AI 사이클보다 유가와 금리의 해석이 우선입니다. 유가 상승이 에너지 주가를 밀어 올리더라도, 동시에 성장주 할인율을 높일 수 있습니다.",
            "따라서 에너지 강세를 시장 전체 위험 선호로 착각하면 안 됩니다.",
        ])
    elif regime["regime"] == "방어주 로테이션":
        rows.extend([
            "오늘은 시장이 수익률보다 안정성을 찾는 장세입니다. 방어 섹터 강세는 지수 하락을 막아줄 수 있지만, 공격적 성장주의 추세 회복을 의미하지는 않습니다.",
        ])
    elif regime["regime"] == "시장 폭 확대":
        rows.extend([
            "오늘은 랠리의 질이 좋아지는지 확인할 수 있는 구간입니다. 중소형주와 후발 섹터까지 참여하면 상승의 지속성이 높아질 수 있습니다.",
        ])
    else:
        rows.extend([
            "오늘은 하나의 큰 테마가 시장 전체를 지배하기보다, 섹터별 자금 이동과 개별 종목 뉴스 반응이 더 중요합니다.",
            "이런 날에는 큰 서사를 과하게 적용하기보다, 실제 가격 반응이 강한 종목만 남기는 것이 더 낫습니다.",
        ])

    smh = macro.get("smh")
    xlk = macro.get("xlk")
    xle = macro.get("xle")
    if smh:
        rows.append(f"반도체 확인값: SMH {smh.change_pct:+.2f}%, 신호 {smh.signal}.")
    if xlk:
        rows.append(f"기술주 확인값: XLK {xlk.change_pct:+.2f}%, 신호 {xlk.signal}.")
    if xle:
        rows.append(f"에너지 확인값: XLE {xle.change_pct:+.2f}%, 신호 {xle.signal}.")
    return block("2. 오늘의 시장 사이클 진단", rows)


def build_break_points(
    market: list[TechnicalSnapshot],
    sectors: list[TechnicalSnapshot],
    stocks: list[TechnicalSnapshot],
) -> str:
    regime = classify_market_regime(market, sectors, stocks)
    macro = regime["macro"]
    rows = []

    if regime["regime"] == "AI/반도체 반등":
        rows.extend([
            "오늘의 반등 가설이 깨지는 조건은 SMH가 상승폭을 반납하거나, AVGO/NVDA 같은 대장주가 종가 기준으로 버티지 못하는 것입니다.",
            "반등이 진짜라면 소프트웨어/클라우드까지 확산되어야 합니다. 반도체만 오르고 빅테크가 약하면 단기 숏커버일 수 있습니다.",
        ])
    elif regime["regime"] == "유가/인플레이션 로테이션":
        rows.extend([
            "유가 로테이션 가설이 깨지는 조건은 유가 상승에도 에너지 섹터가 따라가지 못하거나, 금리가 급등하며 지수 전체가 눌리는 경우입니다.",
            "반대로 유가가 진정되고 TLT가 반등하면 성장주 부담은 빠르게 완화될 수 있습니다.",
        ])
    else:
        rows.extend([
            "현재 가설이 깨지는 조건은 강했던 섹터가 종가에 힘을 잃고, 약했던 섹터가 계속 신저가성 흐름을 만드는 것입니다.",
            "좋은 뉴스에도 주가가 못 오르면 기대 선반영, 나쁜 뉴스에도 버티면 수급 우위로 봅니다.",
        ])

    vixy = macro.get("vixy")
    tlt = macro.get("tlt")
    uup = macro.get("uup")
    if vixy and tlt and uup:
        rows.append(f"오늘의 위험 계기판: VIXY {vixy.change_pct:+.2f}%, TLT {tlt.change_pct:+.2f}%, UUP {uup.change_pct:+.2f}%.")
    rows.append("다음 리포트에서는 이 조건들이 유지됐는지, 아니면 시장 성격이 다시 바뀌었는지를 먼저 비교합니다.")
    return block("3. 무엇을 보면 판단을 바꿀 것인가", rows)


def build_action_plan(screened: list[TechnicalSnapshot], stocks: list[TechnicalSnapshot]) -> str:
    excluded = set(MARKET_TICKERS) | set(SECTOR_ETFS)
    weak = [s for s in stocks if s.ticker not in excluded and s.signal == "리스크 점검"]
    strong = [s for s in stocks if s.ticker not in excluded and s.signal in ("특징적 강세", "선별 관찰")]
    rows = []
    if screened:
        rows.append("오늘의 우선 관찰 후보는 " + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in screened[:5]) + "입니다.")
        rows.append("이 후보들은 즉시 매수 대상이라기보다, 다음 거래일에도 상대강도와 지지선을 유지하는지 확인해야 하는 관찰 대상입니다.")
    if strong:
        rows.append("보유 중이라면 강한 종목은 성급히 줄이기보다, 상승분을 지키는지 확인합니다: " + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in strong[:5]))
    if weak:
        rows.append("주의할 종목은 " + ", ".join(f"{s.ticker}({s.change_pct:+.1f}%)" for s in weak[:5]) + "입니다. 반등이 나와도 거래량과 종가 회복이 없으면 비중 확대 근거가 약합니다.")
    rows.append("오늘의 원칙: 시장 성격이 전일과 달라졌다면, 전일의 결론을 고집하지 말고 섹터 주도권과 종가 위치를 기준으로 판단을 갱신합니다.")
    rows.append("레버리지는 줄이고, 틀렸을 때 계좌가 살아남는 구조를 우선합니다.")
    return block("4. 오늘의 행동 지침", rows)


def send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = textwrap.wrap(message, width=3800, replace_whitespace=False, drop_whitespace=False)
    for chunk in chunks:
        response = requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=30)
        response.raise_for_status()


def print_chat_id() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set in .env")

    response = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=30)
    response.raise_for_status()
    updates = response.json().get("result", [])
    if not updates:
        print("No Telegram updates found. Send any message to your bot first, then retry.")
        return

    for update in updates[-5:]:
        message = update.get("message") or update.get("channel_post") or {}
        chat = message.get("chat") or {}
        if chat.get("id"):
            print(f"chat_id={chat['id']} title={chat.get('title') or chat.get('username') or chat.get('first_name')}")


def run_once() -> None:
    report = build_report()
    send_telegram(report)
    print("Telegram report sent.")


def run_scheduler() -> None:
    run_times = [item.strip() for item in os.getenv("RUN_TIMES", "21:00,23:40,06:20").split(",") if item.strip()]
    for run_time in run_times:
        schedule.every().day.at(run_time, "Asia/Seoul").do(run_once)
        print(f"Scheduled daily report at {run_time} KST.")

    while True:
        schedule.run_pending()
        time.sleep(10)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="AI Research Desk Telegram reporter")
    parser.add_argument("--once", action="store_true", help="send one report immediately")
    parser.add_argument("--print", action="store_true", help="print one report without sending Telegram")
    parser.add_argument("--get-chat-id", action="store_true", help="print recent Telegram chat IDs")
    parser.add_argument("--schedule", action="store_true", help="run scheduled reports")
    args = parser.parse_args()

    load_dotenv()

    if args.get_chat_id:
        print_chat_id()
    elif args.print:
        print(build_report())
    elif args.schedule:
        run_scheduler()
    else:
        run_once()


if __name__ == "__main__":
    main()
