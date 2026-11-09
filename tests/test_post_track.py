import csv
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

import pytest

from scripts import post_track
from scripts.post_track import (
    Listing,
    MarketData,
    PostTrackError,
    PriceSeries,
    PriceUnavailable,
)

# Fake KRX calendar: weekdays, with Wed 2026-11-04 as a market holiday.
HOLIDAYS = {date(2026, 11, 4)}
AS_OF = date(2026, 11, 30)
NAMES = {f"{index:06d}": f"종목{index}" for index in range(1, 13)}
MASTER = {code: Listing(name, "KOSPI") for code, name in NAMES.items()}


def trading_days(start=date(2026, 10, 19), end=AS_OF):
    days, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in HOLIDAYS:
            days.append(day)
        day += timedelta(days=1)
    return days


def closes(base, step, end=AS_OF):
    return {day: base + index * step for index, day in enumerate(trading_days(end=end))}


class FakeMarket:
    def __init__(self, *, calendar_end=AS_OF, missing=(), unavailable=(), step=1.0):
        self.calendar_end = calendar_end
        self.missing = set(missing)
        self.unavailable = set(unavailable)
        self.step = step
        self.price_calls = []
        self.benchmark_calls = 0

    def prices(self, code, start, end):
        self.price_calls.append(code)
        if code in self.unavailable:
            raise PriceUnavailable("fake")
        series = closes(100.0, self.step, end=self.calendar_end)
        return PriceSeries("fake", {d: v for d, v in series.items() if d not in self.missing})

    def benchmark(self, start, end):
        self.benchmark_calls += 1
        return PriceSeries("fake-index", closes(1000.0, 5.0, end=self.calendar_end))

    def data(self):
        return MarketData(self.prices, self.benchmark, lambda: MASTER)


def item(ticker="000001", action="buy", name=None):
    code_name = name if name is not None else NAMES.get(ticker, "대상")
    return {"stock_name": code_name, "ticker": ticker, "action": action, "reason": "근거 문장"}


def strategy(items, created_at="2026-11-02T03:00:00Z", **extra):
    return {"schema_version": 2, "strategy_id": f"s-{created_at}-{len(items)}", "report_id": "r-1",
            "strategy_items": items, "created_at": created_at, **extra}


def enough(extra=()):
    """Ten trackable items keep the stop rule out of the way of the unit under test."""
    filler = [item(code, "watch") for code in list(NAMES)[2:12]]
    return [strategy([*extra, *filler])]


def rows_for(records, market=None, as_of=AS_OF, **kwargs):
    market = market or FakeMarket()
    return post_track.build(records, market.data(), as_of, **kwargs)


def test_entry_is_next_trading_day_and_horizons_skip_weekend_and_holiday():
    # Mon 11-02 12:00 KST -> Tue 11-03 entry; Wed 11-04 is closed.
    rows, _ = rows_for(enough([item()]))
    row = rows[0]
    assert row["entry_date"] == "2026-11-03"
    assert row["h1_date"] == "2026-11-05"
    assert row["h5_date"] == "2026-11-11"
    assert row["h10_date"] == "2026-11-18"


@pytest.mark.parametrize(("created_at", "entry"), [
    ("2026-10-31T03:00:00Z", "2026-11-02"),  # Saturday -> Monday
    ("2026-11-02T14:59:00Z", "2026-11-03"),  # Mon 23:59 KST
    ("2026-11-02T15:00:00Z", "2026-11-05"),  # Tue 00:00 KST; next open skips Wed holiday
    ("2026-11-03T22:00:00Z", "2026-11-05"),  # generated on the holiday, KST
])
def test_t0_uses_kst_date_and_excludes_same_day_close(created_at, entry):
    rows, _ = rows_for([strategy([item()], created_at), *enough()])
    assert rows[0]["entry_date"] == entry


def test_returns_benchmark_excess_and_direction():
    rows, _ = rows_for(enough([item("000001", "buy"), item("000002", "sell")]), FakeMarket(step=2.0))
    buy, sell = rows[0], rows[1]
    entry_index = trading_days().index(date(2026, 11, 3))
    entry, later = 100.0 + 2 * entry_index, 100.0 + 2 * (entry_index + 1)
    index_entry, index_later = 1000.0 + 5 * entry_index, 1000.0 + 5 * (entry_index + 1)
    expected = round((later / entry - 1) * 100, 4)
    expected_index = round((index_later / index_entry - 1) * 100, 4)
    assert buy["entry_close"] == entry
    assert buy["h1_return_pct"] == expected
    assert buy["h1_kospi_return_pct"] == expected_index
    assert buy["h1_excess_pct"] == round(expected - expected_index, 4)
    assert (buy["h1_direction"], sell["h1_direction"]) == ("match", "mismatch")


def test_hold_and_watch_show_returns_without_direction():
    rows, summary = rows_for(enough([item("000001", "hold")]))
    assert rows[0]["h1_status"] == "ok"
    assert rows[0]["h1_direction"] == ""
    assert summary["horizons"]["1"]["buy_sell_counted"] == 0


def test_flat_return_is_neither_match_nor_mismatch():
    rows, summary = rows_for(enough([item()]), FakeMarket(step=0.0))
    assert rows[0]["h1_direction"] == "flat"
    assert summary["horizons"]["1"]["direction_flat"] == 1


def test_undue_horizons_stay_blank():
    rows, summary = rows_for(enough([item()]), FakeMarket(calendar_end=date(2026, 11, 12)),
                             as_of=date(2026, 11, 12))
    row = rows[0]
    assert row["h1_status"] == row["h5_status"] == "ok"
    assert row["h10_status"] == "not_due"
    assert not any(key in row for key in ("h10_date", "h10_return_pct", "h10_close"))
    assert summary["horizons"]["10"] == {
        "ok": 0, "not_due": 11, "price_missing": 0, "buy_sell_counted": 0,
        "direction_match": 0, "direction_mismatch": 0, "direction_flat": 0,
    }


def test_entry_not_reached_yields_no_values():
    late = date(2026, 11, 2)
    rows, _ = rows_for(enough([item()]), FakeMarket(calendar_end=late), as_of=late)
    assert rows[0]["status"] == "entry_pending"
    assert "entry_date" not in rows[0]
    assert all(rows[0][f"h{n}_status"] == "not_due" for n in (1, 5, 10))


def test_closes_after_as_of_are_ignored_even_if_provider_returns_them():
    as_of = date(2026, 11, 12)  # the fake still returns data through 11-30
    rows, summary = rows_for(enough([item()]), as_of=as_of)
    assert rows[0]["h10_status"] == "not_due"
    assert summary["last_trading_day"] == "2026-11-12"


def test_no_prices_for_a_ticker_is_reported_without_values():
    market = FakeMarket(unavailable={"000001"})
    rows, _ = rows_for(enough([item()]), market)
    assert rows[0]["status"] == "price_missing"
    assert "h1_return_pct" not in rows[0]
    assert all(row["status"] == "tracked" for row in rows[1:])


def test_missing_entry_close_blanks_the_whole_row():
    market = FakeMarket(missing={date(2026, 11, 3)})
    rows, _ = rows_for(enough([item()]), market)
    assert rows[0]["status"] == "price_missing"
    assert "entry_close" not in rows[0]


def test_halted_horizon_day_is_blank_while_others_compute():
    market = FakeMarket(missing={date(2026, 11, 5)})
    rows, summary = rows_for(enough([item()]), market)
    assert rows[0]["h1_status"] == "price_missing"
    assert "h1_return_pct" not in rows[0]
    assert rows[0]["h5_status"] == "ok"
    assert summary["horizons"]["1"]["price_missing"] == 11


def test_missing_ticker_is_excluded_with_reason_and_fill_ratio():
    market = FakeMarket()
    rows, summary = rows_for(enough([item(None, name="정유 업종"), item("  ", name="공백")]), market)
    assert [row["exclusion_reason"] for row in rows[:2]] == ["ticker_missing", "ticker_missing"]
    assert all(row["status"] == "excluded" for row in rows[:2])
    assert summary["items_total"] == 12
    assert summary["ticker_filled"] == 10
    assert summary["ticker_fill_ratio"] == round(10 / 12, 4)
    assert summary["excluded_reasons"] == {"ticker_missing": 2}
    assert len(market.price_calls) == 10


@pytest.mark.parametrize(("raw", "reason"), [
    ("NVDA", "ticker_not_krx_format"), (5930, "ticker_not_krx_format"), ("999999", "not_in_master"),
])
def test_invalid_and_unlisted_tickers_are_counted(raw, reason):
    rows, summary = rows_for(enough([item(raw, name="대상")]))
    assert rows[0]["exclusion_reason"] == reason
    assert summary["excluded_reasons"] == {reason: 1}


@pytest.mark.parametrize("raw", ["000001.KS", " a000001", "KRX:000001", "000001.KQ"])
def test_common_ticker_spellings_normalize(raw):
    assert post_track.normalize_ticker(raw) == "000001"


def test_name_mismatch_is_flagged_and_kept():
    rows, summary = rows_for(enough([item("000001", name="다른회사")]))
    assert rows[0]["name_check"] == "mismatch"
    assert rows[0]["status"] == "tracked"
    assert summary["name_mismatch"] == 1


def test_below_threshold_holds_without_touching_prices():
    market = FakeMarket()
    rows, summary = rows_for([strategy([item(code) for code in list(NAMES)[:9]])], market)
    assert summary["hold"]["verdict"] == "hold"
    assert summary["hold"]["minimum_trackable"] == 10
    assert market.price_calls == [] and market.benchmark_calls == 0
    assert all(row["status"] == "on_hold" and "h1_return_pct" not in row for row in rows)


def test_exactly_threshold_proceeds():
    _, summary = rows_for([strategy([item(code) for code in list(NAMES)[:10]])])
    assert summary["hold"]["verdict"] == "proceed"


def test_empty_export_holds_with_undefined_ratio():
    _, summary = rows_for([])
    assert summary["ticker_fill_ratio"] is None
    assert summary["hold"]["verdict"] == "hold"


def test_coverage_only_never_looks_up_prices():
    market = FakeMarket()
    rows, summary = rows_for(enough([item()]), market, coverage_only=True)
    assert summary["hold"]["verdict"] == "proceed"
    assert market.price_calls == [] and market.benchmark_calls == 0
    assert rows[0]["status"] == "coverage_only"


def test_rule_filters_are_counted_and_start_date_uses_kst():
    records = [
        strategy([item()], "2026-10-18T15:00:00Z"),  # 10-19 00:00 KST: in
        strategy([item()], "2026-10-18T14:59:59Z"),  # 10-18 23:59 KST: out
        {**strategy([item()]), "schema_version": 1},
        {**strategy([item()]), "created_at": "not-a-date"},
        "not-an-object",
        *enough(),
    ]
    records.append(deepcopy(records[0]))
    rows, summary = rows_for(records)
    assert summary["strategies_skipped"] == {
        "before_rule_start": 1, "duplicate_strategy_id": 1, "invalid_created_at": 1,
        "invalid_record": 1, "not_schema_v2": 1,
    }
    assert summary["strategies_in_rule"] == 2
    assert len(rows) == 11


@pytest.mark.parametrize("value", [
    "2026-11-02T03:00:00Z", "2026-11-02T03:00:00", {"$date": "2026-11-02T03:00:00.000Z"},
    {"$date": 1793588400000}, {"$date": {"$numberLong": "1793588400000"}},
    datetime(2026, 11, 2, 3, 0),
])
def test_created_at_formats(value):
    assert post_track.parse_created_at(value) == datetime(2026, 11, 2, 3, 0, tzinfo=UTC)


def test_rule_hash_is_pinned_so_changes_are_deliberate():
    assert post_track.rule_sha256() == "50f51a06d81047548eef49910ea33aea051a3167d40617f3934ae47cf7224dc9"


def test_input_records_are_not_mutated():
    records = enough([item(), item(None, name="업종")])
    before = deepcopy(records)
    rows_for(records)
    assert records == before


def test_default_as_of_waits_for_settled_close():
    before = datetime(2026, 11, 2, 6, 59, tzinfo=UTC)  # 15:59 KST
    after = datetime(2026, 11, 2, 7, 0, tzinfo=UTC)  # 16:00 KST
    assert post_track.default_as_of(before) == date(2026, 11, 1)
    assert post_track.default_as_of(after) == date(2026, 11, 2)


def test_first_available_falls_back_and_reports_sources():
    def broken():
        raise RuntimeError("provider detail")

    closes_map = {date(2026, 11, 2): 10.0}
    assert post_track.first_available(("a", broken), ("b", lambda: closes_map)) == ("b", closes_map)
    with pytest.raises(PriceUnavailable) as caught:
        post_track.first_available(("a", broken), ("b", dict))
    assert str(caught.value) == "a:RuntimeError,b:empty"


def test_main_writes_csv_with_notice_and_leaves_input_unchanged(tmp_path, capsys):
    source = tmp_path / "strategies.json"
    source.write_text(json.dumps(enough([item(), item(None, name="업종")]), ensure_ascii=False), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "out" / "track.csv"
    code = post_track.main([str(source), "--output", str(output), "--as-of", "2026-11-12"],
                           market=FakeMarket().data(), now=datetime(2026, 11, 30, 9, tzinfo=UTC))
    assert code == 0
    assert source.read_bytes() == original
    printed = capsys.readouterr().out
    assert "투자 성과 입증이 아닌" in printed
    assert "ticker 채움: 11/12" in printed
    assert "김건 확정 필요" in printed
    with output.open(encoding="utf-8-sig", newline="") as file:
        table = list(csv.DictReader(file))
    assert len(table) == 12
    assert table[0]["h10_return_pct"] == ""
    assert table[0]["h1_return_pct"] != ""
    assert table[1]["exclusion_reason"] == "ticker_missing"
    assert all("투자 성과 입증이 아닌" in row["notice"] for row in table)
    summary = json.loads(output.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["rule_sha256"] == post_track.rule_sha256()


def test_main_rejects_unsettled_as_of_and_overwriting_input(tmp_path):
    source = tmp_path / "strategies.json"
    source.write_text("[]", encoding="utf-8")
    market = FakeMarket().data()
    now = datetime(2026, 11, 2, 6, 0, tzinfo=UTC)
    with pytest.raises(SystemExit) as unsettled:
        post_track.main([str(source), "--output", str(tmp_path / "o.csv"), "--as-of", "2026-11-02"],
                        market=market, now=now)
    assert unsettled.value.code == 2
    with pytest.raises(SystemExit) as overwrite:
        post_track.main([str(source), "--output", str(source)], market=market, now=now)
    assert overwrite.value.code == 2
    assert source.read_text(encoding="utf-8") == "[]"


def test_load_records_accepts_mongoexport_shapes(tmp_path):
    ndjson = tmp_path / "a.json"
    ndjson.write_text('{"strategy_items": []}\n{"strategy_items": []}\n', encoding="utf-8")
    wrapped = tmp_path / "b.json"
    wrapped.write_text('{"strategies": [{"strategy_items": []}]}', encoding="utf-8")
    assert len(post_track.load_records(ndjson)) == 2
    assert len(post_track.load_records(wrapped)) == 1
    bad = tmp_path / "c.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(PostTrackError):
        post_track.load_records(bad)
