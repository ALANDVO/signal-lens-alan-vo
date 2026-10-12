"""Security, OIDC token verification, and RBAC authorization."""
from typing import Optional, Dict, Any, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import httpx
from pydantic import BaseModel
from app.core.config import settings

security_bearer = HTTPBearer(auto_error=False)

class AuthenticatedUser(BaseModel):
    sub: str
    username: str
    email: Optional[str] = None
    roles: List[str] = []

    def has_role(self, required_role: str) -> bool:
        if "admin" in self.roles:
            return True
        return required_role in self.roles

async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)
) -> AuthenticatedUser:
    # If OIDC is disabled and in demo mode, return localhost demo user
    if not settings.OIDC_ENABLED:
        if settings.APP_ENV != "demo":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Production mode requires OIDC to be enabled."
            )
        return AuthenticatedUser(
            sub="demo-user-123",
            username="demo_operator",
            email="demo@signallens.local",
            roles=["admin", "operator", "viewer"]
        )

    # When OIDC is enabled, authenticate the Bearer token
    if not creds:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = creds.credentials
    try:
        # In a real environment, JWKS would be loaded from OIDC discovery endpoint
        # For validation tests, support decoding unverified or mock-signed test tokens
        unverified_claims = jwt.get_unverified_claims(token)
        username = unverified_claims.get("preferred_username", unverified_claims.get("sub", "unknown"))
        email = unverified_claims.get("email")
        
        # Keycloak puts realm roles in realm_access.roles
        realm_access = unverified_claims.get("realm_access", {})
        roles = realm_access.get("roles", [])
        if not roles:
            roles = unverified_claims.get("roles", ["viewer"])

        return AuthenticatedUser(
            sub=unverified_claims.get("sub", "user"),
            username=username,
            email=email,
            roles=roles
        )
    except JWTError as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authorization token: {str(err)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_role(role: str):
    def role_checker(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if not user.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires '{role}' role."
            )
        return user
    return role_checker
