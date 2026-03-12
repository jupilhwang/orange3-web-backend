"""
mDNS Service Discovery Module.

This module provides mDNS (Multicast DNS) based service advertisement
for the Orange3 Web Backend. It allows the backend to advertise itself
on the local network so that frontends can automatically discover it.

Configuration (orange3-web-backend.properties):
    mdns.enabled=true
    mdns.service_type=_orange3-web._tcp
    mdns.service_name=orange3-backend
    mdns.multicast_address=224.0.0.251
    mdns.udp_port=5353
    mdns.interface=

Note: The service port is automatically taken from server.port config.
"""

import socket
import ipaddress
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Check zeroconf availability
try:
    from zeroconf import ServiceInfo, Zeroconf, IPVersion, InterfaceChoice
    from zeroconf.asyncio import AsyncZeroconf

    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False
    # Define stub classes for type hints when zeroconf is not available
    ServiceInfo = None
    Zeroconf = None
    IPVersion = None
    AsyncZeroconf = None

    # Create a stub InterfaceChoice enum
    class InterfaceChoice:
        All = "all"
        Default = "default"

    logger.warning("zeroconf not installed. mDNS service discovery disabled.")
    logger.warning("Install with: pip install zeroconf")


# Default mDNS settings (RFC 6762, IPv4 only)
DEFAULT_SERVICE_TYPE = "_orange3-web._tcp.local."
DEFAULT_MULTICAST_ADDRESS = "224.0.0.251"
DEFAULT_UDP_PORT = 5353


@dataclass
class MDNSConfig:
    """mDNS configuration (IPv4 only)."""

    # Service settings
    enabled: bool = True
    service_type: str = DEFAULT_SERVICE_TYPE
    service_name: str = "orange3-backend"
    port: int = 8000  # Service port (set from server.port at runtime)

    # Network settings (IPv4 only)
    multicast_address: str = DEFAULT_MULTICAST_ADDRESS
    udp_port: int = DEFAULT_UDP_PORT
    interface: str = ""  # Empty = all interfaces

    # TXT record properties (metadata)
    properties: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config_manager(
        cls, config_manager, server_port: int = 8000
    ) -> "MDNSConfig":
        """Create MDNSConfig from ConfigManager.

        Args:
            config_manager: Configuration manager instance
            server_port: Server port from server.port config (passed at runtime)
        """
        # Get service_type and ensure it ends with ".local."
        service_type = config_manager.get("mdns.service_type", "_orange3-web._tcp")
        if not service_type.endswith(".local."):
            if service_type.endswith("."):
                service_type = service_type + "local."
            else:
                service_type = service_type + ".local."

        return cls(
            enabled=config_manager.get("mdns.enabled", True, bool),
            service_type=service_type,
            service_name=config_manager.get("mdns.service_name", "orange3-backend"),
            port=server_port,  # Use server.port instead of mdns.port
            multicast_address=config_manager.get(
                "mdns.multicast_address", DEFAULT_MULTICAST_ADDRESS
            ),
            udp_port=config_manager.get("mdns.udp_port", DEFAULT_UDP_PORT, int),
            interface=config_manager.get("mdns.interface", ""),
        )


class MDNSService:
    """
    mDNS service advertiser.

    Advertises the backend service on the local network using mDNS.
    Frontends can discover this service and register it with their
    load balancer.

    Usage:
        config = MDNSConfig(service_name="orange3-backend", port=8000)
        service = MDNSService(config)

        # Register service
        await service.register()

        # ... run application ...

        # Unregister on shutdown
        await service.unregister()
    """

    def __init__(self, config: MDNSConfig):
        """
        Initialize mDNS service.

        Args:
            config: mDNS configuration
        """
        self.config = config
        self._zeroconf: Optional[AsyncZeroconf] = None
        self._service_info: Optional[ServiceInfo] = None
        self._registered = False

    def _get_interface_ipv4s(self, interface_name: str) -> List[str]:
        """Get non-loopback IPv4 addresses for the named NIC interface.

        Uses ifaddr (pure Python) to query OS network interface info directly,
        without DNS resolution.

        Args:
            interface_name: NIC name (e.g. "eth0", "en0")

        Returns:
            List of non-loopback IPv4 address strings. Empty list if not found.
        """
        try:
            import ifaddr
        except ImportError:
            logger.warning(
                "ifaddr not installed, cannot bind to specific interface. "
                "Install with: pip install ifaddr"
            )
            return []

        result: List[str] = []
        for adapter in ifaddr.get_adapters():
            if adapter.name == interface_name:
                for ip_info in adapter.ips:
                    if not ip_info.is_IPv4:
                        continue
                    try:
                        if not ipaddress.ip_address(ip_info.ip).is_loopback:
                            result.append(ip_info.ip)
                    except ValueError:
                        logger.debug(
                            f"Skipping malformed IP from ifaddr: {ip_info.ip!r}"
                        )
        return result

    def _get_local_ip(self) -> str:
        """Get local IP address for the service.

        If mdns.interface is configured, returns the IP of that NIC.
        Otherwise, determines the default outgoing IP via UDP socket.
        """
        if self.config.interface:
            try:
                ips = self._get_interface_ipv4s(self.config.interface)
                if ips:
                    logger.info(f"Using interface {self.config.interface} IP: {ips[0]}")
                    return ips[0]
                logger.warning(
                    f"Interface {self.config.interface} not found or has no IPv4 address"
                )
            except Exception as e:
                logger.warning(
                    f"Could not get IP for interface {self.config.interface}: {e}"
                )

        # Default: determine local IP via outgoing UDP socket (no actual data sent)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.1)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            except Exception as e:
                logger.debug(f"Suppressed error: {e}")
                ip = "127.0.0.1"
            finally:
                s.close()
            return ip
        except Exception as e:
            logger.warning(f"Could not determine local IP: {e}")
            return "127.0.0.1"

    def _get_hostname(self) -> str:
        """Get hostname for service name."""
        try:
            hostname = socket.gethostname()
            # Remove domain part if present
            if "." in hostname:
                hostname = hostname.split(".")[0]
            return hostname
        except Exception as e:
            logger.debug(f"Suppressed error: {e}")
            return "unknown"

    def _build_service_name(self) -> str:
        """Build unique service name with hostname."""
        base_name = self.config.service_name

        # Replace {hostname} placeholder
        if "{hostname}" in base_name:
            hostname = self._get_hostname()
            return base_name.replace("{hostname}", hostname)

        return base_name

    def _get_interfaces(self) -> Optional[List[str]]:
        """Get list of interface IPs for zeroconf binding.

        Returns:
            List of IP addresses for the configured interface,
            None to bind on all interfaces.
        """
        if not self.config.interface:
            return None

        try:
            ips = self._get_interface_ipv4s(self.config.interface)
            if ips:
                return ips
            logger.warning(
                f"Interface {self.config.interface} not found or has no IPv4 address, "
                "using all interfaces"
            )
        except Exception as e:
            logger.warning(f"Could not get interface IPs: {e}, using all interfaces")

        return None

    async def register(self, properties: Optional[Dict[str, str]] = None) -> bool:
        """
        Register the service with mDNS.

        Args:
            properties: Additional TXT record properties

        Returns:
            True if registration successful, False otherwise
        """
        if not ZEROCONF_AVAILABLE:
            logger.warning("mDNS not available (zeroconf not installed)")
            return False

        if not self.config.enabled:
            logger.info("mDNS is disabled in configuration")
            return False

        if self._registered:
            logger.warning("mDNS service already registered")
            return True

        try:
            # Build service name
            service_name = self._build_service_name()

            # Get local IP
            local_ip = self._get_local_ip()

            # Merge properties
            txt_properties = {**self.config.properties}
            if properties:
                txt_properties.update(properties)

            # Convert properties to bytes for TXT record
            txt_record = {
                k: v.encode("utf-8") if isinstance(v, str) else str(v).encode("utf-8")
                for k, v in txt_properties.items()
            }

            # Create ServiceInfo
            self._service_info = ServiceInfo(
                type_=self.config.service_type,
                name=f"{service_name}.{self.config.service_type}",
                port=self.config.port,
                properties=txt_record,
                server=f"{service_name}.local.",
                addresses=[socket.inet_aton(local_ip)],
            )

            # Create Zeroconf instance (IPv4 only)
            # Note: Custom multicast address/port requires patching zeroconf internals
            # For standard mDNS, we use the default settings
            interfaces = self._get_interfaces()
            if interfaces:
                logger.info(f"Using specific interfaces for mDNS: {interfaces}")
                self._zeroconf = AsyncZeroconf(
                    ip_version=IPVersion.V4Only, interfaces=interfaces
                )
            else:
                self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)

            # Register service
            await self._zeroconf.async_register_service(self._service_info)

            self._registered = True

            logger.info(
                f"✅ mDNS service registered: {service_name} "
                f"({local_ip}:{self.config.port})"
            )
            logger.info(f"   Service type: {self.config.service_type}")
            logger.info(f"   TXT records: {txt_properties}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to register mDNS service: {e}")
            return False

    async def unregister(self) -> bool:
        """
        Unregister the service from mDNS.

        Returns:
            True if unregistration successful, False otherwise
        """
        if not self._registered:
            return True

        try:
            if self._zeroconf and self._service_info:
                await self._zeroconf.async_unregister_service(self._service_info)
                await self._zeroconf.async_close()

            self._registered = False
            self._zeroconf = None
            self._service_info = None

            logger.info("✅ mDNS service unregistered")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to unregister mDNS service: {e}")
            return False

    @property
    def is_registered(self) -> bool:
        """Check if service is registered."""
        return self._registered


# Global mDNS service instance
_mdns_service: Optional[MDNSService] = None


def get_mdns_service() -> Optional[MDNSService]:
    """Get the global mDNS service instance."""
    return _mdns_service


def set_mdns_service(service: MDNSService):
    """Set the global mDNS service instance."""
    global _mdns_service
    _mdns_service = service


@asynccontextmanager
async def mdns_lifespan(
    config: MDNSConfig, properties: Optional[Dict[str, str]] = None
):
    """
    Async context manager for mDNS service lifecycle.

    Usage:
        async with mdns_lifespan(config, {"version": "1.0.0"}):
            # Service is registered
            await run_app()
        # Service is automatically unregistered

    Args:
        config: mDNS configuration
        properties: TXT record properties
    """
    service = MDNSService(config)
    set_mdns_service(service)

    try:
        await service.register(properties)
        yield service
    finally:
        await service.unregister()


def is_mdns_available() -> bool:
    """Check if mDNS is available."""
    return ZEROCONF_AVAILABLE
