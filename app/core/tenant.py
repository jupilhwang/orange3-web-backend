"""
Multi-tenant management for Orange3 Web Backend
"""
from typing import Dict, List, Optional
from fastapi import Header, HTTPException
import uuid

from .models import Tenant


class TenantManager:
    """Manages tenants in the multi-tenant system."""
    
    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        # Create a default tenant
        default_tenant = Tenant(
            id="default",
            name="Default Tenant"
        )
        self._tenants["default"] = default_tenant
    
    def create_tenant(self, name: str) -> Tenant:
        """Create a new tenant."""
        tenant = Tenant(name=name)
        self._tenants[tenant.id] = tenant
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        return list(self._tenants.values())
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            return True
        return False


# Global tenant manager instance
_tenant_manager = TenantManager()


async def get_current_tenant(
    x_tenant_id: str = Header(default="default", alias="X-Tenant-ID")
) -> Tenant:
    """
    Dependency to get the current tenant from the request header.
    
    The tenant ID should be passed in the X-Tenant-ID header.
    Defaults to "default" tenant if not specified.
    """
    tenant = _tenant_manager.get_tenant(x_tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant '{x_tenant_id}' not found"
        )
    return tenant


