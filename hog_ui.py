from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlparse

from hog_backtest_service import BacktestRequest, aggregate_request_from_daily_series
from hog_price_baseline import (
    DEFAULT_CACHE_PATH,
    DEFAULT_PURCHASE_TYPE,
    HogObservation,
    download_direct_hog_history,
    load_cached_daily_series,
)


REPO_ROOT = Path(__file__).resolve().parent
APP_ASSET_DIR = REPO_ROOT / "cloudflare" / "ui-worker" / "public" / "app"
LOGO_PATH = REPO_ROOT / "boss-hog-logo.png"
ASSET_ROUTES: dict[str, Path] = {
    "/": APP_ASSET_DIR / "index.html",
    "/index.html": APP_ASSET_DIR / "index.html",
    "/app/styles.css": APP_ASSET_DIR / "styles.css",
    "/app/app.js": APP_ASSET_DIR / "app.js",
    "/app/logo.png": LOGO_PATH,
}
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
}


def resolve_asset(route_path: str) -> tuple[Path, str] | None:
    asset_path = ASSET_ROUTES.get(route_path)
    if asset_path is None:
        return None
    content_type = CONTENT_TYPES.get(asset_path.suffix)
    if content_type is None:
        raise ValueError(f"Unsupported asset type for {asset_path}.")
    return asset_path, content_type


def run_local_backtest(raw_params: Mapping[str, object]) -> dict[str, object]:
    request = BacktestRequest.from_mapping(raw_params)
    cache_path = download_direct_hog_history(
        DEFAULT_CACHE_PATH,
        purchase_type=DEFAULT_PURCHASE_TYPE,
    )
    daily_series = load_cached_daily_series(cache_path)
    return aggregate_request_from_daily_series(
        daily_series,
        request,
        purchase_type=DEFAULT_PURCHASE_TYPE,
        source="USDA AMS direct hog avg_net_price",
        refreshed_at=_timestamp_from_path(cache_path),
    )


class HogUIHandler(BaseHTTPRequestHandler):
    server_version = "BossHogUI/0.2"

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request(include_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_request(include_body=False)

    def _handle_request(self, *, include_body: bool) -> None:
        parsed = urlparse(self.path)

        asset = resolve_asset(parsed.path)
        if asset is not None:
            self._serve_file(asset[0], asset[1], include_body=include_body)
            return

        if parsed.path == "/api/health":
            self._serve_json({"ok": True, "service": "boss-hog-local-ui"}, include_body=include_body)
            return

        if parsed.path == "/api/backtest":
            self._serve_backtest(parsed.query, include_body=include_body)
            return

        self.send_error(404, "Not found")

    def _serve_backtest(self, query: str, *, include_body: bool) -> None:
        try:
            payload = run_local_backtest(parse_qs(query))
        except ValueError as error:
            self._serve_json({"error": str(error)}, status=400, include_body=include_body)
            return
        except Exception as error:  # pragma: no cover - exercised manually
            print(json.dumps({"event": "local_backtest_error", "message": str(error)}))
            self._serve_json({"error": "Backtest run failed."}, status=500, include_body=include_body)
            return
        self._serve_json(payload, include_body=include_body)

    def _serve_file(self, path: Path, content_type: str, *, include_body: bool) -> None:
        if not path.exists():
            self.send_error(404, "Not found")
            return
        body = path.read_bytes()
        self._send_response(
            status=200,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store" if path.suffix != ".png" else "public, max-age=3600",
            },
            body=body,
            include_body=include_body,
        )

    def _serve_json(self, payload: Mapping[str, object], *, status: int = 200, include_body: bool) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_response(
            status=status,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
                "Cache-Control": "no-store",
            },
            body=body,
            include_body=include_body,
        )

    def _send_response(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        body: bytes,
        include_body: bool,
    ) -> None:
        try:
            self.send_response(status)
            for header_name, header_value in headers.items():
                self.send_header(header_name, header_value)
            self.end_headers()
            if include_body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Boss Hog UI using the shared Worker frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), HogUIHandler)
    print(f"Boss Hog UI available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _timestamp_from_path(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
