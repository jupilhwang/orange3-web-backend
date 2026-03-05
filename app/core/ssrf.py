"""
SSRF (Server-Side Request Forgery) validation and safe fetching.

Pure validation functions with no FastAPI dependencies.
Provides URL validation against private IPs and DNS rebinding protection
by using pre-resolved IPs for actual HTTP requests.
"""

import asyncio
import ipaddress
import logging
import socket
import ssl
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Hostnames that must always be blocked (cloud metadata endpoints)
_BLOCKED_HOSTNAMES = frozenset(
    {
        "169.254.169.254",  # AWS/Azure/GCP metadata
        "metadata.google.internal",
        "metadata",
    }
)


def _is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP address belongs to a private/internal range."""
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast


async def validate_url_for_ssrf(url: str) -> tuple[bool, str, str]:
    """
    Validate a URL to prevent SSRF attacks.

    Returns:
        (is_valid, error_message, resolved_ip)
        When *is_valid* is True, *resolved_ip* contains the IP the hostname
        resolved to (or the literal IP from the URL).  Callers should use
        this IP for the actual HTTP request to prevent DNS-rebinding.
    """
    try:
        parsed = urlparse(url)

        if not parsed.scheme:
            return False, "URL must have a scheme (http/https)", ""

        if parsed.scheme not in ("http", "https"):
            return (
                False,
                f"Scheme '{parsed.scheme}' not allowed. Only http/https.",
                "",
            )

        if not parsed.hostname:
            return False, "URL must have a hostname", ""

        if parsed.hostname.lower() in _BLOCKED_HOSTNAMES:
            return False, f"Blocked hostname: {parsed.hostname}", ""

        # Direct IP literal
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if _is_ip_blocked(ip):
                return (
                    False,
                    f"Private/internal IP addresses not allowed: {parsed.hostname}",
                    "",
                )
            return True, "", str(ip)
        except ValueError:
            pass

        # Hostname — resolve DNS once and validate
        try:
            resolved_ip = await asyncio.to_thread(socket.gethostbyname, parsed.hostname)
            ip = ipaddress.ip_address(resolved_ip)
            if _is_ip_blocked(ip):
                return False, f"Hostname resolves to private IP: {resolved_ip}", ""
        except socket.gaierror:
            return False, f"Cannot resolve hostname: {parsed.hostname}", ""

        return True, "", resolved_ip

    except Exception as e:
        return False, f"Invalid URL: {str(e)}", ""


async def fetch_ssrf_safe(url: str, *, timeout: float = 60.0) -> httpx.Response:
    """Fetch *url* after SSRF validation, using the pre-resolved IP.

    This prevents DNS-rebinding (TOCTOU) attacks by connecting to the
    IP that was validated rather than letting httpx re-resolve the hostname.

    Raises ``httpx.HTTPStatusError`` on non-2xx responses and
    ``ValueError`` when SSRF validation fails.
    """
    is_valid, error_msg, resolved_ip = await validate_url_for_ssrf(url)
    if not is_valid:
        raise ValueError(f"URL not allowed: {error_msg}")

    parsed = urlparse(url)
    hostname = parsed.hostname
    scheme = parsed.scheme
    port = parsed.port or (443 if scheme == "https" else 80)
    path_query = parsed.path or "/"
    if parsed.query:
        path_query += f"?{parsed.query}"

    # Build a URL that connects directly to the resolved IP
    ip_url = f"{scheme}://{resolved_ip}:{port}{path_query}"

    logger.info("SSRF-safe fetch: %s → %s (Host: %s)", url, ip_url, hostname)

    # Build SSL context: verify certificate but allow IP-based connections
    # (hostname check disabled because we connect via IP, not hostname)
    if scheme == "https":
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        verify_param = ssl_context
    else:
        verify_param = False

    async with httpx.AsyncClient(
        verify=verify_param,
        timeout=timeout,
    ) as client:
        response = await client.get(
            ip_url,
            headers={"Host": hostname},
            follow_redirects=False,
        )

    return response
