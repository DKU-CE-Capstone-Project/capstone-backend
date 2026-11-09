"""Build the 1/5/10 trading-day post-hoc tracking table from exported v2 strategies.

This is a small-sample observation record, not evidence of investment performance.
It reads a local JSON export only and never connects to MongoDB or the service.
Market data needs the optional extra: pip install ".[post-track]"

Typical order, so that the stop decision is taken before any return is seen:
    python scripts/post_track.py strategies.json --output track.csv --coverage-only
    python scripts/post_track.py strategies.json --output track.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from math import isfinite
from pathlib import Path

KST = timezone(timedelta(hours=9))
MAX_INPUT_BYTES = 20_000_000
# Closing auction ends 15:30 KST; before this a source may return an intraday "close".
MARKET_SETTLED_KST = time(16, 0)

# Fixed before results are seen. There is deliberately no CLI override: changing the
# rule means changing this code, which also changes the printed rule hash.
# Source: econmind-docs docs/18-backtest-design-review.md 3.1, 4.3, 6.1. The horizon
# 10 replaces the document's 20 per docs 25 section 3 (1/5/10 trading days).
TARGET_RULE = {
    "schema_version": 2,
    "created_from_kst": "2026-10-19",
    "universe": "strategy_items whose ticker matches the domestic listing master",
    "t0": "strategy created_at, in KST",
    "entry": "close of the first trading day strictly after the t0 KST date",
    "horizons": [1, 5, 10],
    "benchmark": "KOSPI",
    "min_trackable_items": 10,
}
HORIZONS = tuple(TARGET_RULE["horizons"])
CREATED_FROM = date.fromisoformat(TARGET_RULE["created_from_kst"])
MIN_TRACKABLE_ITEMS = TARGET_RULE["min_trackable_items"]

NOTICE = (
    "투자 성과 입증이 아닌 소표본 사후 추적 관찰 기록입니다. 통계적 유의성이 없고 "
    "거래 비용·세금·슬리피지를 반영하지 않았습니다. 연구·교육용이며 투자 권유가 아니고, "
    "사후 추적 결과는 미래 수익을 보장하지 않습니다."
)
HOLD_CRITERION = (
    "추적 가능 건수가 기준 미만이면 보류(문서 18 4.3·6.1). "
    "기준 10건은 문서의 예시값이며 팀 합의 미확인, 김건 확정 필요."
)
OPEN_DECISIONS = (
    "대상 시작일 2026-10-19는 문서 18 6.1의 예시값이며 팀 결정 미확인, 김건 확정 필요",
    "보류 기준 10건은 문서 18의 예시값이며 팀 합의 미확인, 김건 확정 필요",
    "name_check=mismatch 행은 규칙상 유지하며 수익률을 보기 전에 사람이 확인",
    "원천 시세의 수정주가 여부와 이용 약관은 사람이 확인(문서 18 2.2, 5.2)",
)


class PostTrackError(ValueError):
    """A fixed error code; never carries the supplied document."""


class PriceUnavailable(Exception):
    pass


@dataclass(frozen=True)
class PriceSeries:
    source: str
    closes: dict[date, float]


@dataclass(frozen=True)
class Listing:
    name: str
    market: str


@dataclass(frozen=True)
class MarketData:
    """Injectable market access: prices(code, start, end), benchmark(start, end), master()."""

    prices: Callable[[str, date, date], PriceSeries]
    benchmark: Callable[[date, date], PriceSeries]
    master: Callable[[], dict[str, Listing]]


def rule_sha256() -> str:
    raw = json.dumps(TARGET_RULE, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def default_as_of(now: datetime) -> date:
    local = now.astimezone(KST)
    return local.date() if local.time() >= MARKET_SETTLED_KST else local.date() - timedelta(days=1)


def parse_created_at(value) -> datetime | None:
    """Accept BSON datetimes, ISO strings and mongoexport Extended JSON. Naive means UTC."""
    if isinstance(value, dict):
        value = value.get("$date")
        if isinstance(value, dict):
            value = value.get("$numberLong")
            value = int(value) if isinstance(value, str) and value.lstrip("-").isdigit() else None
        if isinstance(value, int) and not isinstance(value, bool):
            return datetime.fromtimestamp(value / 1000, UTC)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def normalize_ticker(value) -> str | None:
    """Return a canonical KRX code (six characters; new listings may contain letters)."""
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    text = re.sub(r"^(KRX|KOSPI|KOSDAQ|KONEX):", "", text)
    text = re.sub(r"\.(KS|KQ)$", "", text)
    text = re.sub(r"^A(?=\d{6}$)", "", text)
    return text if re.fullmatch(r"\d[0-9A-Z]{5}", text) else None


def _name_key(value) -> str:
    text = value if isinstance(value, str) else ""
    return re.sub(r"\s+|\(주\)|주식회사", "", text).casefold()


def load_records(path: Path) -> list:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise PostTrackError("INPUT_FILE_TOO_LARGE")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError:
        raise PostTrackError("INVALID_INPUT_ENCODING") from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:  # mongoexport writes one document per line unless --jsonArray is given.
            value = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            raise PostTrackError("INVALID_INPUT_JSON") from None
    if isinstance(value, dict):
        value = value.get("strategies", [value])
    if not isinstance(value, list):
        raise PostTrackError("EXPECTED_STRATEGY_LIST")
    return value


def _select(records: list) -> tuple[list[tuple[dict, datetime]], Counter]:
    selected, skipped, seen = [], Counter(), set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("strategy_items"), list):
            skipped["invalid_record"] += 1
            continue
        if record.get("schema_version") != TARGET_RULE["schema_version"]:
            skipped["not_schema_v2"] += 1
            continue
        created = parse_created_at(record.get("created_at"))
        if created is None:
            skipped["invalid_created_at"] += 1
            continue
        if created.astimezone(KST).date() < CREATED_FROM:
            skipped["before_rule_start"] += 1
            continue
        strategy_id = record.get("strategy_id")
        if strategy_id is not None and strategy_id in seen:
            skipped["duplicate_strategy_id"] += 1
            continue
        seen.add(strategy_id)
        selected.append((record, created))
    return selected, skipped


def _classify(item, master: dict[str, Listing]) -> tuple[dict, str | None]:
    data = item if isinstance(item, dict) else {}
    raw = data.get("ticker")
    row = {"stock_name": data.get("stock_name") or "", "ticker_raw": "" if raw is None else raw,
           "ticker": "", "master_name": "", "market": "", "name_check": "",
           "action": data.get("action") or ""}
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return row, "ticker_missing"
    code = normalize_ticker(raw)
    if code is None:
        return row, "ticker_not_krx_format"
    row["ticker"] = code
    listing = master.get(code)
    if listing is None:
        return row, "not_in_master"
    row.update(master_name=listing.name, market=listing.market,
               name_check="match" if _name_key(row["stock_name"]) == _name_key(listing.name)
               else "mismatch")
    return row, None


def _ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def _pct(start: float, end: float) -> float:
    return round((end / start - 1) * 100, 4)


def _direction(action: str, value: float) -> str:
    if action not in {"buy", "sell"}:
        return ""  # hold/watch show returns only and stay out of the direction count.
    if value == 0:
        return "flat"
    return "match" if (value > 0) == (action == "buy") else "mismatch"


def _settled(series: PriceSeries, as_of: date) -> dict[date, float]:
    # Enforced here so no provider, real or injected, can leak a later close.
    return {day: close for day, close in series.closes.items() if day <= as_of and close > 0}


def _fill_returns(row: dict, t0: date, calendar: list[date], kospi: dict[date, float],
                  series: PriceSeries | None) -> None:
    entry_index = bisect_right(calendar, t0)
    if entry_index >= len(calendar):
        row["status"] = "entry_pending"
        for horizon in HORIZONS:
            row[f"h{horizon}_status"] = "not_due"
        return
    entry_date = calendar[entry_index]
    row["entry_date"] = entry_date.isoformat()
    closes = series.closes if series else {}
    entry_close = closes.get(entry_date)
    if entry_close is None:
        row["status"] = "price_missing"
        for horizon in HORIZONS:
            row[f"h{horizon}_status"] = "price_missing"
        return
    row.update(status="tracked", entry_close=entry_close, price_source=series.source)
    for horizon in HORIZONS:
        prefix, target = f"h{horizon}_", entry_index + horizon
        if target >= len(calendar):
            row[prefix + "status"] = "not_due"
            continue
        day = calendar[target]
        row[prefix + "date"] = day.isoformat()
        if day not in closes:
            row[prefix + "status"] = "price_missing"
            continue
        stock = _pct(entry_close, closes[day])
        index = _pct(kospi[entry_date], kospi[day])
        row.update({prefix + "status": "ok", prefix + "close": closes[day],
                    prefix + "return_pct": stock, prefix + "kospi_return_pct": index,
                    prefix + "excess_pct": round(stock - index, 4),
                    prefix + "direction": _direction(row["action"], stock)})


def build(records: list, market: MarketData, as_of: date,
          *, coverage_only: bool = False) -> tuple[list[dict], dict]:
    selected, skipped = _select(records)
    master = market.master()
    if not master:
        raise PostTrackError("LISTING_MASTER_UNAVAILABLE")
    rows, reasons = [], Counter()
    for record, created in selected:
        for index, item in enumerate(record["strategy_items"]):
            row, reason = _classify(item, master)
            local = created.astimezone(KST)
            row.update(strategy_id=record.get("strategy_id") or "",
                       report_id=record.get("report_id") or "", item_index=index,
                       created_at_kst=local.isoformat(), _t0=local.date(), exclusion_reason=reason or "",
                       status="excluded" if reason else "")
            reasons.update([reason] if reason else [])
            rows.append(row)

    total = len(rows)
    trackable = [row for row in rows if not row["exclusion_reason"]]
    filled = total - reasons["ticker_missing"]
    hold = len(trackable) < MIN_TRACKABLE_ITEMS
    summary = {
        "notice": NOTICE, "rule": TARGET_RULE, "rule_sha256": rule_sha256(),
        "as_of": as_of.isoformat(), "mode": "coverage_only" if coverage_only else "tracking",
        "strategies_in_file": len(records), "strategies_in_rule": len(selected),
        "strategies_skipped": dict(sorted(skipped.items())),
        "items_total": total, "ticker_filled": filled, "ticker_fill_ratio": _ratio(filled, total),
        "trackable": len(trackable), "trackable_ratio": _ratio(len(trackable), total),
        "excluded_reasons": dict(sorted(reasons.items())),
        "name_mismatch": sum(row["name_check"] == "mismatch" for row in rows),
        "hold": {"verdict": "hold" if hold else "proceed", "minimum_trackable": MIN_TRACKABLE_ITEMS,
                 "criterion": HOLD_CRITERION},
        "open_decisions": list(OPEN_DECISIONS),
        "last_trading_day": None, "benchmark_source": None, "horizons": {},
    }
    if hold or coverage_only:
        for row in trackable:
            row["status"] = "on_hold" if hold else "coverage_only"
        return _finish(rows, summary)

    start = min(row["_t0"] for row in trackable)
    try:
        benchmark = market.benchmark(start, as_of)
    except PriceUnavailable:
        raise PostTrackError("BENCHMARK_UNAVAILABLE") from None
    kospi = _settled(benchmark, as_of)
    calendar = sorted(kospi)
    if not calendar:
        raise PostTrackError("BENCHMARK_UNAVAILABLE")

    fetched: dict[str, PriceSeries | None] = {}
    for row in trackable:
        code = row["ticker"]
        if code not in fetched:
            try:
                series = market.prices(code, start, as_of)
                fetched[code] = PriceSeries(series.source, _settled(series, as_of))
            except PriceUnavailable:
                fetched[code] = None
        row["benchmark_source"] = benchmark.source
        _fill_returns(row, row["_t0"], calendar, kospi, fetched[code])

    summary.update(last_trading_day=calendar[-1].isoformat(), benchmark_source=benchmark.source)
    for horizon in HORIZONS:
        states = Counter(row[f"h{horizon}_status"] for row in trackable)
        directions = Counter(row.get(f"h{horizon}_direction", "") for row in trackable)
        directions.pop("", None)
        summary["horizons"][str(horizon)] = {
            "ok": states["ok"], "not_due": states["not_due"], "price_missing": states["price_missing"],
            "buy_sell_counted": sum(directions.values()), "direction_match": directions["match"],
            "direction_mismatch": directions["mismatch"], "direction_flat": directions["flat"],
        }
    return _finish(rows, summary)


def _finish(rows: list[dict], summary: dict) -> tuple[list[dict], dict]:
    for row in rows:
        del row["_t0"]
        row.update(rule_sha256=summary["rule_sha256"], notice=NOTICE)
    return rows, summary


def csv_fields() -> list[str]:
    fields = ["strategy_id", "report_id", "item_index", "created_at_kst", "stock_name", "ticker_raw",
              "ticker", "master_name", "market", "name_check", "action", "status", "exclusion_reason",
              "entry_date", "entry_close"]
    for horizon in HORIZONS:
        fields += [f"h{horizon}_{name}" for name in
                   ("status", "date", "close", "return_pct", "kospi_return_pct", "excess_pct", "direction")]
    return fields + ["price_source", "benchmark_source", "rule_sha256", "notice"]


def write_outputs(rows: list[dict], summary: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields(), restval="")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = summary_path_for(output)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def summary_path_for(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def render_summary(summary: dict) -> str:
    def share(value: float | None) -> str:
        return "산출 불가(항목 0건)" if value is None else f"{value:.1%}"

    lines = [
        f"[고지] {summary['notice']}",
        f"대상 규칙 해시: {summary['rule_sha256']}",
        (f"데이터 기준일: {summary['as_of']} / 마지막 거래일: {summary['last_trading_day'] or '조회 안 함'} "
         f"/ 기준 지수 출처: {summary['benchmark_source'] or '조회 안 함'}"),
        (f"전략 {summary['strategies_in_file']}건 중 규칙 해당 {summary['strategies_in_rule']}건, "
         f"제외 {summary['strategies_skipped']}"),
        f"ticker 채움: {summary['ticker_filled']}/{summary['items_total']} ({share(summary['ticker_fill_ratio'])})",
        f"추적 가능: {summary['trackable']}/{summary['items_total']} ({share(summary['trackable_ratio'])})",
        f"항목 제외 사유: {summary['excluded_reasons']}",
        f"이름 불일치(유지·사람 확인): {summary['name_mismatch']}건",
    ]
    hold = summary["hold"]
    verdict = "보류" if hold["verdict"] == "hold" else "진행"
    lines.append(f"판정: {verdict} (추적 가능 {summary['trackable']}건, 기준 {hold['minimum_trackable']}건). {hold['criterion']}")
    if hold["verdict"] == "hold":
        lines.append("보류 판정이라 시세를 조회하지 않았고 수익률을 만들지 않았습니다.")
    elif summary["mode"] == "coverage_only":
        lines.append("--coverage-only 실행이라 시세를 조회하지 않았습니다.")
    for horizon, stats in summary["horizons"].items():
        lines.append(
            f"{horizon}거래일: 값 있음 {stats['ok']}, 미도래 {stats['not_due']}, 시세 없음 {stats['price_missing']} "
            f"/ buy·sell 방향 일치 {stats['direction_match']}/{stats['buy_sell_counted']} (보합 {stats['direction_flat']})"
        )
    lines.append("미확정 항목:")
    lines += [f"- {item}" for item in summary["open_decisions"]]
    return "\n".join(lines)


def _closes(series, start: date, end: date) -> dict[date, float]:
    result = {}
    for stamp, value in series.items():
        day = stamp.date() if hasattr(stamp, "date") else stamp
        close = float(value)
        if start <= day <= end and isfinite(close) and close > 0:
            result[day] = close
    return result


def first_available(*sources: tuple[str, Callable[[], dict]]) -> tuple[str, dict]:
    """Return the first source's (name, non-empty data); a fallback never mixes sources."""
    failures = []
    for name, load in sources:
        try:
            closes = load()
        except Exception as exc:  # noqa: BLE001 - third-party scrapers raise arbitrary types
            failures.append(f"{name}:{type(exc).__name__}")
            continue
        if closes:
            return name, closes
        failures.append(f"{name}:empty")
    raise PriceUnavailable(",".join(failures))


def _fdr_close(symbol: str, start: date, end: date) -> dict[date, float]:
    import FinanceDataReader as fdr

    return _closes(fdr.DataReader(symbol, start.isoformat(), end.isoformat())["Close"], start, end)


def _pykrx_stock_close(code: str, start: date, end: date) -> dict[date, float]:
    from pykrx import stock

    frame = stock.get_market_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), code, adjusted=True)
    return _closes(frame["종가"], start, end)


def _pykrx_index_close(start: date, end: date) -> dict[date, float]:
    from pykrx import stock

    frame = stock.get_index_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), "1001")
    return _closes(frame["종가"], start, end)


def _fdr_master() -> dict[str, Listing]:
    import FinanceDataReader as fdr

    frame = fdr.StockListing("KRX")
    return {str(row.Code): Listing(str(row.Name), str(row.Market)) for row in frame.itertuples()}


def _pykrx_master(day: date) -> dict[str, Listing]:
    from pykrx import stock

    stamp = day.strftime("%Y%m%d")
    return {
        code: Listing(stock.get_market_ticker_name(code), market)
        for market in ("KOSPI", "KOSDAQ", "KONEX")
        for code in stock.get_market_ticker_list(stamp, market=market)
    }


def load_master_csv(path: Path) -> dict[str, Listing]:
    """Read a hand-downloaded listing with columns code,name,market."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["code"].strip(): Listing(row["name"].strip(), row["market"].strip())
                for row in csv.DictReader(file)}


def default_market_data(as_of: date, master_csv: Path | None = None) -> MarketData:
    def prices(code: str, start: date, end: date) -> PriceSeries:
        return PriceSeries(*first_available(("FinanceDataReader", lambda: _fdr_close(code, start, end)),
                                            ("pykrx", lambda: _pykrx_stock_close(code, start, end))))

    def benchmark(start: date, end: date) -> PriceSeries:
        return PriceSeries(*first_available(("FinanceDataReader", lambda: _fdr_close("KS11", start, end)),
                                            ("pykrx", lambda: _pykrx_index_close(start, end))))

    def master() -> dict[str, Listing]:
        if master_csv is not None:
            return load_master_csv(master_csv)
        try:
            return first_available(("FinanceDataReader", _fdr_master),
                                   ("pykrx", lambda: _pykrx_master(as_of)))[1]
        except PriceUnavailable:
            return {}

    return MarketData(prices, benchmark, master)


def main(argv: list[str] | None = None, *, market: MarketData | None = None,
         now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="local strategies export (JSON array, NDJSON or {strategies:[...]})")
    parser.add_argument("--output", type=Path, required=True, help="CSV path; a .summary.json is written next to it")
    parser.add_argument("--as-of", type=date.fromisoformat, help="last settled KST date to use (default: latest settled)")
    parser.add_argument("--coverage-only", action="store_true", help="report ticker coverage and the stop verdict without any price lookup")
    parser.add_argument("--master-csv", type=Path, help="local listing CSV (code,name,market) instead of the online listing")
    args = parser.parse_args(argv)

    settled = default_as_of(now or datetime.now(UTC))
    as_of = args.as_of or settled
    if as_of > settled:
        parser.exit(2, "AS_OF_NOT_SETTLED\n")
    if args.input.resolve() in {args.output.resolve(), summary_path_for(args.output).resolve()}:
        parser.exit(2, "OUTPUT_WOULD_OVERWRITE_INPUT\n")
    try:
        records = load_records(args.input)
        rows, summary = build(records, market or default_market_data(as_of, args.master_csv),
                              as_of, coverage_only=args.coverage_only)
        summary_path = write_outputs(rows, summary, args.output)
    except PostTrackError as exc:
        parser.exit(2, f"{exc}\n")
    except OSError:
        parser.exit(2, "POST_TRACK_FILE_IO_FAILED\n")
    print(render_summary(summary))
    print(f"CSV: {args.output}\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
