"""
Reverse proxy header middleware.

Processes X-Forwarded-* headers from trusted reverse proxies
and strips them from untrusted sources.
"""

import ipaddress
import os

from fastapi import FastAPI


class ProxyHeadersMiddleware:
    """Process X-Forwarded-* headers from trusted reverse proxies."""

    TRUSTED_CIDRS = [
        ipaddress.ip_network(cidr.strip())
        for cidr in os.environ.get(
            "TRUSTED_PROXIES",
            "127.0.0.0/8,::1/128,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
        ).split(",")
        if cidr.strip()
    ]

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client")
            if client:
                try:
                    client_ip = ipaddress.ip_address(client[0])
                except ValueError:
                    client_ip = None
                is_trusted = client_ip is not None and any(
                    client_ip in network for network in self.TRUSTED_CIDRS
                )

                headers = dict(scope.get("headers", []))
                if is_trusted:
                    xff = headers.get(b"x-forwarded-for", b"").decode()
                    if xff:
                        real_ip = xff.split(",")[0].strip()
                        scope["client"] = (real_ip, client[1])
                else:
                    filtered_headers = [
                        (k, v)
                        for k, v in scope.get("headers", [])
                        if k.lower()
                        not in (
                            b"x-forwarded-for",
                            b"x-forwarded-proto",
                            b"x-forwarded-host",
                            b"x-real-ip",
                        )
                    ]
                    scope["headers"] = filtered_headers

        await self.app(scope, receive, send)


def setup_proxy_middleware(app: FastAPI) -> None:
    """Register the proxy headers middleware on the FastAPI app."""
    app.add_middleware(ProxyHeadersMiddleware)
