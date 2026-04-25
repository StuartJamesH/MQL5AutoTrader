"""
data.py — Fetch OHLCV trendbar data from the cTrader Open API.

NOTE — Twisted reactor limitation
    reactor.run() is blocking and can only be called ONCE per Python process.
    Calling fetch_ohlcv() repeatedly in the same process will raise a
    ReactorNotRestartable error.  For multi-symbol downloads, use
    fetch_ohlcv_bulk(), which fetches all requested symbols inside a single
    authenticated client session and a single reactor run.
"""

from __future__ import annotations

import calendar
import datetime
import os
import pathlib
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
from twisted.internet import reactor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIOD_MAP: dict[str, int] = {
    "M1":  ProtoOATrendbarPeriod.M1,
    "M2":  ProtoOATrendbarPeriod.M2,
    "M3":  ProtoOATrendbarPeriod.M3,
    "M4":  ProtoOATrendbarPeriod.M4,
    "M5":  ProtoOATrendbarPeriod.M5,
    "M10": ProtoOATrendbarPeriod.M10,
    "M15": ProtoOATrendbarPeriod.M15,
    "M30": ProtoOATrendbarPeriod.M30,
    "H1":  ProtoOATrendbarPeriod.H1,
    "H4":  ProtoOATrendbarPeriod.H4,
    "H12": ProtoOATrendbarPeriod.H12,
    "D1":  ProtoOATrendbarPeriod.D1,
    "W1":  ProtoOATrendbarPeriod.W1,
    "MN1": ProtoOATrendbarPeriod.MN1,
}

# Path to the shared data/ directory (two levels above this file)
_DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_period(period_str: str) -> int:
    if period_str not in PERIOD_MAP:
        raise ValueError(
            f"Unknown period '{period_str}'. Valid options: {list(PERIOD_MAP.keys())}"
        )
    return PERIOD_MAP[period_str]


def _load_credentials() -> dict[str, Any]:
    load_dotenv()
    return {
        "ClientId":    os.getenv("CLIENT_ID"),
        "Secret":      os.getenv("SECRET"),
        "HostType":    os.getenv("HOST_TYPE"),
        "AccessToken": os.getenv("ACCESS_TOKEN"),
        "AccountId":   int(os.getenv("ACCOUNT_ID")),
    }


def _build_client(credentials: dict[str, Any]) -> Client:
    host = (
        EndPoints.PROTOBUF_LIVE_HOST
        if credentials["HostType"].lower() == "live"
        else EndPoints.PROTOBUF_DEMO_HOST
    )
    return Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)


def _transform_trendbar(trendbar) -> list[Any]:
    open_time = datetime.datetime.fromtimestamp(
        trendbar.utcTimestampInMinutes * 60, datetime.timezone.utc
    )
    open_price = (trendbar.low + trendbar.deltaOpen) / 100000.0
    high_price = (trendbar.low + trendbar.deltaHigh) / 100000.0
    low_price = trendbar.low / 100000.0
    close_price = (trendbar.low + trendbar.deltaClose) / 100000.0
    return [open_time, open_price, high_price, low_price, close_price, trendbar.volume]


def _build_trendbar_requests(
    *,
    symbol_id: int,
    account_id: int,
    bar_period: int,
    num_chunks: int,
    weeks_per_chunk: int,
) -> list[Any]:
    now = datetime.datetime.now(datetime.timezone.utc)
    requests: list[Any] = []
    for i in range(num_chunks):
        to_time = now - datetime.timedelta(weeks=weeks_per_chunk * i)
        from_time = to_time - datetime.timedelta(weeks=weeks_per_chunk)
        req = ProtoOAGetTrendbarsReq()
        req.symbolId = symbol_id
        req.ctidTraderAccountId = account_id
        req.period = bar_period
        req.fromTimestamp = int(calendar.timegm(from_time.utctimetuple())) * 1000
        req.toTimestamp = int(calendar.timegm(to_time.utctimetuple())) * 1000
        requests.append(req)
    return requests


def _build_ohlcv_dataframe(bars: list[list[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        np.array(bars),
        columns=["Time", "Open", "High", "Low", "Close", "Volume"],
    ).drop_duplicates().reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])

    return df.sort_values("Time").drop_duplicates().reset_index(drop=True)


def _default_output_path(
    *,
    symbol_name: str,
    period_str: str,
    num_chunks: int,
    weeks_per_chunk: int,
) -> str:
    total_weeks = num_chunks * weeks_per_chunk
    return str(_DATA_DIR / f"{symbol_name}_{period_str}_{total_weeks}weeks.csv")


def _save_ohlcv_csv(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def _fetch_ohlcv_bulk_data(
    *,
    symbol_names: list[str],
    num_chunks: int,
    weeks_per_chunk: int,
    period_str: str,
) -> dict[str, pd.DataFrame]:
    if not symbol_names:
        raise ValueError("symbol_names must contain at least one symbol.")

    credentials = _load_credentials()
    bar_period = _validate_period(period_str)
    client = _build_client(credentials)
    results: dict[str, pd.DataFrame] = {}
    fetch_error: Exception | None = None

    def _record_error(error: Any) -> None:
        nonlocal fetch_error
        exc = getattr(error, "value", error)
        if isinstance(exc, Exception):
            fetch_error = exc
        else:
            fetch_error = RuntimeError(str(exc))
        print("\nMessage Error:", exc)
        if reactor.running:
            reactor.stop()

    def _on_error(failure: Any) -> None:
        _record_error(failure)

    def _on_message_received(_client: Client, message: Any) -> None:
        silent_types = {
            ProtoHeartbeatEvent().payloadType,
            ProtoOAAccountAuthRes().payloadType,
            ProtoOAApplicationAuthRes().payloadType,
            ProtoOASymbolsListRes().payloadType,
            ProtoOAGetTrendbarsRes().payloadType,
        }
        if message.payloadType in silent_types:
            return
        print("\nMessage received:\n", Protobuf.extract(message))

    def _disconnected(_client: Client, reason: Any) -> None:
        print("\nDisconnected:", reason)

    def _symbols_response_callback(result: Any) -> None:
        try:
            print("\nSymbols received")
            symbols_response = Protobuf.extract(result)
            symbol_lookup: dict[str, Any] = {}
            for symbol_name in symbol_names:
                matches = [s for s in symbols_response.symbol if s.symbolName == symbol_name]
                if len(matches) == 0:
                    raise ValueError(f"No symbol matches '{symbol_name}'")
                if len(matches) > 1:
                    raise ValueError(f"Multiple symbols match '{symbol_name}': {matches}")
                symbol_lookup[symbol_name] = matches[0]

            def _fetch_symbol(symbol_index: int) -> None:
                if symbol_index >= len(symbol_names):
                    print("\nAll symbols fetched")
                    if reactor.running:
                        reactor.stop()
                    return

                symbol_name = symbol_names[symbol_index]
                symbol = symbol_lookup[symbol_name]
                requests = _build_trendbar_requests(
                    symbol_id=symbol.symbolId,
                    account_id=credentials["AccountId"],
                    bar_period=bar_period,
                    num_chunks=num_chunks,
                    weeks_per_chunk=weeks_per_chunk,
                )
                daily_bars: list[list[Any]] = []
                print(f"\nFetching symbol {symbol_index + 1}/{len(symbol_names)}: {symbol_name}")

                def _fetch_chunk(chunk_index: int) -> None:
                    if chunk_index >= len(requests):
                        results[symbol_name] = _build_ohlcv_dataframe(daily_bars)
                        print(f"Completed {symbol_name}: {len(results[symbol_name])} rows")
                        _fetch_symbol(symbol_index + 1)
                        return

                    deferred = client.send(requests[chunk_index])

                    def _on_success(chunk_result: Any) -> None:
                        trendbars = Protobuf.extract(chunk_result)
                        bars_data = list(map(_transform_trendbar, trendbars.trendbar))
                        daily_bars.extend(bars_data)
                        print(
                            f"\nFetched {symbol_name} chunk {chunk_index + 1}/{len(requests)}, "
                            f"bars: {len(bars_data)}"
                        )
                        _fetch_chunk(chunk_index + 1)

                    deferred.addCallbacks(_on_success, _on_error)

                _fetch_chunk(0)

            _fetch_symbol(0)
        except Exception as exc:  # pragma: no cover - callback error path
            _record_error(exc)

    def _account_auth_response_callback(_result: Any) -> None:
        print("\nAccount authenticated")
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId = credentials["AccountId"]
        req.includeArchivedSymbols = False
        deferred = client.send(req)
        deferred.addCallbacks(_symbols_response_callback, _on_error)

    def _application_auth_response_callback(_result: Any) -> None:
        print("\nApplication authenticated")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = credentials["AccountId"]
        req.accessToken = credentials["AccessToken"]
        deferred = client.send(req)
        deferred.addCallbacks(_account_auth_response_callback, _on_error)

    def _connected(_client: Client) -> None:
        print("\nConnected")
        req = ProtoOAApplicationAuthReq()
        req.clientId = credentials["ClientId"]
        req.clientSecret = credentials["Secret"]
        deferred = client.send(req)
        deferred.addCallbacks(_application_auth_response_callback, _on_error)

    client.setConnectedCallback(_connected)
    client.setDisconnectedCallback(_disconnected)
    client.setMessageReceivedCallback(_on_message_received)

    client.startService()
    reactor.run()

    if fetch_error is not None:
        raise fetch_error

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    symbol_name: str,
    num_chunks: int,
    weeks_per_chunk: int,
    period_str: str,
    save_csv: bool = False,
    output_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV trendbar data for a single symbol from the cTrader Open API."""
    data_by_symbol = _fetch_ohlcv_bulk_data(
        symbol_names=[symbol_name],
        num_chunks=num_chunks,
        weeks_per_chunk=weeks_per_chunk,
        period_str=period_str,
    )
    df = data_by_symbol[symbol_name]

    if save_csv:
        final_output_path = output_path or _default_output_path(
            symbol_name=symbol_name,
            period_str=period_str,
            num_chunks=num_chunks,
            weeks_per_chunk=weeks_per_chunk,
        )
        _save_ohlcv_csv(df, final_output_path)
        return None

    return df


def fetch_ohlcv_bulk(
    symbol_names: Iterable[str],
    num_chunks: int,
    weeks_per_chunk: int,
    period_str: str,
    save_csv: bool = False,
) -> Optional[dict[str, pd.DataFrame]]:
    """Fetch OHLCV trendbar data for multiple symbols in one reactor run.

    Parameters
    ----------
    symbol_names : Iterable[str]
        Symbol names exactly as they appear in the broker's symbol list.
    num_chunks : int
        Number of time chunks to fetch per symbol.
    weeks_per_chunk : int
        Width of each chunk in weeks.
    period_str : str
        Bar period string. One of the keys in :data:`PERIOD_MAP`.
    save_csv : bool, optional
        When ``True``, write one CSV per symbol using the same naming pattern
        as :func:`fetch_ohlcv` and return ``None``. Otherwise return a
        ``dict[str, pandas.DataFrame]`` keyed by symbol.
    """
    normalized_symbols = [str(symbol) for symbol in symbol_names]
    data_by_symbol = _fetch_ohlcv_bulk_data(
        symbol_names=normalized_symbols,
        num_chunks=num_chunks,
        weeks_per_chunk=weeks_per_chunk,
        period_str=period_str,
    )

    if save_csv:
        for symbol_name, df in data_by_symbol.items():
            output_path = _default_output_path(
                symbol_name=symbol_name,
                period_str=period_str,
                num_chunks=num_chunks,
                weeks_per_chunk=weeks_per_chunk,
            )
            _save_ohlcv_csv(df, output_path)
        return None

    return data_by_symbol


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = fetch_ohlcv(
        symbol_name="US500",
        num_chunks=52 * 10,
        weeks_per_chunk=1,
        period_str="M1",
        save_csv=True,
    )
