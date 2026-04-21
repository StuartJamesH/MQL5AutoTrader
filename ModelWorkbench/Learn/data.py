"""
data.py — Fetch OHLCV trendbar data from the cTrader Open API.

NOTE — Twisted reactor limitation
    reactor.run() is blocking and can only be called ONCE per Python process.
    Calling fetch_ohlcv() a second time in the same process will raise a
    ReactorNotRestartable error.  This is acceptable for a data-download
    utility script run from the command line or from a freshly-started
    notebook kernel.  If multi-call support is needed, migrate to the
    asyncio-based cTrader API instead.
"""

from __future__ import annotations

import calendar
import datetime
import os
import pathlib
from typing import Optional

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
    """Fetch OHLCV trendbar data from the cTrader Open API.

    Parameters
    ----------
    symbol_name : str
        Symbol name exactly as it appears in the broker's symbol list
        (e.g. ``"US500"``, ``"EURUSD"``).
    num_chunks : int
        Number of time chunks to fetch.
    weeks_per_chunk : int
        Width of each chunk in weeks.
    period_str : str
        Bar period string.  One of: M1 M2 M3 M4 M5 M10 M15 M30 H1 H4 H12 D1
        W1 MN1.
    save_csv : bool, optional
        When ``False`` (default) a :class:`pandas.DataFrame` is returned.
        When ``True`` the data is written to a CSV file and ``None`` is
        returned.
    output_path : str | None, optional
        Path for the CSV file when ``save_csv=True``.  If ``None``, the path
        is auto-generated as
        ``<repo_root>/data/{symbol_name}_{period_str}_{total_weeks}weeks.csv``.

    Returns
    -------
    pandas.DataFrame or None
        The OHLCV data as a DataFrame when ``save_csv=False``, otherwise
        ``None``.

    Raises
    ------
    ValueError
        If ``period_str`` is not in :data:`PERIOD_MAP`.
    """
    load_dotenv()

    if period_str not in PERIOD_MAP:
        raise ValueError(
            f"Unknown period '{period_str}'. Valid options: {list(PERIOD_MAP.keys())}"
        )

    # ── Credentials & client ─────────────────────────────────────────────────
    credentials = {
        "ClientId":    os.getenv("CLIENT_ID"),
        "Secret":      os.getenv("SECRET"),
        "HostType":    os.getenv("HOST_TYPE"),
        "AccessToken": os.getenv("ACCESS_TOKEN"),
        "AccountId":   int(os.getenv("ACCOUNT_ID")),
    }

    host = (
        EndPoints.PROTOBUF_LIVE_HOST
        if credentials["HostType"].lower() == "live"
        else EndPoints.PROTOBUF_DEMO_HOST
    )
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

    bar_period = PERIOD_MAP[period_str]
    daily_bars: list = []

    # ── Trendbar transform ───────────────────────────────────────────────────
    def _transform_trendbar(trendbar):
        open_time   = datetime.datetime.fromtimestamp(
            trendbar.utcTimestampInMinutes * 60, datetime.timezone.utc
        )
        open_price  = (trendbar.low + trendbar.deltaOpen)  / 100000.0
        high_price  = (trendbar.low + trendbar.deltaHigh)  / 100000.0
        low_price   =  trendbar.low                        / 100000.0
        close_price = (trendbar.low + trendbar.deltaClose) / 100000.0
        return [open_time, open_price, high_price, low_price, close_price, trendbar.volume]

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _on_error(failure):
        print("\nMessage Error:", failure)

    def _on_message_received(client, message):
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

    def _disconnected(client, reason):
        print("\nDisconnected:", reason)

    def _symbols_response_callback(result):
        nonlocal daily_bars
        print("\nSymbols received")
        symbols = Protobuf.extract(result)
        matches = [s for s in symbols.symbol if s.symbolName == symbol_name]
        if len(matches) == 0:
            raise Exception(f"No symbol matches '{symbol_name}'")
        if len(matches) > 1:
            raise Exception(f"Multiple symbols match '{symbol_name}': {matches}")
        symbol = matches[0]

        now = datetime.datetime.utcnow()
        requests = []
        for i in range(num_chunks):
            to_time   = now - datetime.timedelta(weeks=weeks_per_chunk * i)
            from_time = to_time - datetime.timedelta(weeks=weeks_per_chunk)
            req = ProtoOAGetTrendbarsReq()
            req.symbolId            = symbol.symbolId
            req.ctidTraderAccountId = credentials["AccountId"]
            req.period              = bar_period
            req.fromTimestamp       = int(calendar.timegm(from_time.utctimetuple())) * 1000
            req.toTimestamp         = int(calendar.timegm(to_time.utctimetuple()))   * 1000
            requests.append(req)

        daily_bars.clear()

        def _fetch_next(index):
            if index >= len(requests):
                print("\nAll chunks fetched")
                reactor.stop()
                return
            deferred = client.send(requests[index])

            def _on_success(result):
                nonlocal daily_bars
                trendbars = Protobuf.extract(result)
                bars_data = list(map(_transform_trendbar, trendbars.trendbar))
                daily_bars.extend(bars_data)
                print(f"\nFetched chunk {index + 1}/{len(requests)}, bars: {len(bars_data)}")
                _fetch_next(index + 1)

            deferred.addCallbacks(_on_success, _on_error)

        _fetch_next(0)

    def _account_auth_response_callback(result):
        print("\nAccount authenticated")
        req = ProtoOASymbolsListReq()
        req.ctidTraderAccountId    = credentials["AccountId"]
        req.includeArchivedSymbols = False
        deferred = client.send(req)
        deferred.addCallbacks(_symbols_response_callback, _on_error)

    def _application_auth_response_callback(result):
        print("\nApplication authenticated")
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = credentials["AccountId"]
        req.accessToken         = credentials["AccessToken"]
        deferred = client.send(req)
        deferred.addCallbacks(_account_auth_response_callback, _on_error)

    def _connected(client):
        print("\nConnected")
        req = ProtoOAApplicationAuthReq()
        req.clientId     = credentials["ClientId"]
        req.clientSecret = credentials["Secret"]
        deferred = client.send(req)
        deferred.addCallbacks(_application_auth_response_callback, _on_error)

    # ── Wire up and run ──────────────────────────────────────────────────────
    client.setConnectedCallback(_connected)
    client.setDisconnectedCallback(_disconnected)
    client.setMessageReceivedCallback(_on_message_received)

    client.startService()
    reactor.run()

    # ── Build DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame(
        np.array(daily_bars),
        columns=["Time", "Open", "High", "Low", "Close", "Volume"],
    ).drop_duplicates().reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("Time").drop_duplicates().reset_index(drop=True)

    # ── Return or save ───────────────────────────────────────────────────────
    if save_csv:
        if output_path is None:
            total_weeks = num_chunks * weeks_per_chunk
            output_path = str(
                _DATA_DIR / f"{symbol_name}_{period_str}_{total_weeks}weeks.csv"
            )
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")
        return None

    return df


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
