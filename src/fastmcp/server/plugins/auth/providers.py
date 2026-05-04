"""First-party auth plugins.

These plugins are thin, JSON-configurable wrappers around FastMCP's
existing auth providers. Python-only dependencies such as HTTP clients,
token verifiers, and client storage stay as constructor arguments.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

import httpx
from key_value.aio.protocols import AsyncKeyValue
from pydantic import AnyHttpUrl, BaseModel, ConfigDict

from fastmcp.server.auth import AuthProvider, TokenVerifier
from fastmcp.server.auth.providers.auth0 import Auth0Provider
from fastmcp.server.auth.providers.aws import AWSCognitoProvider
from fastmcp.server.auth.providers.azure import AzureProvider
from fastmcp.server.auth.providers.clerk import ClerkProvider
from fastmcp.server.auth.providers.descope import DescopeProvider
from fastmcp.server.auth.providers.discord import DiscordProvider
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.auth.providers.keycloak import KeycloakAuthProvider
from fastmcp.server.auth.providers.oci import OCIProvider
from fastmcp.server.auth.providers.propelauth import (
    PropelAuthProvider,
    PropelAuthTokenIntrospectionOverrides,
)
from fastmcp.server.auth.providers.scalekit import ScalekitProvider
from fastmcp.server.auth.providers.supabase import SupabaseProvider
from fastmcp.server.auth.providers.workos import AuthKitProvider, WorkOSProvider
from fastmcp.server.plugins.base import Plugin, PluginMeta

ConsentMode = bool | Literal["remember", "external"]
Algorithm = Literal["RS256", "ES256"]
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class _AuthPlugin(Plugin[ConfigT], Generic[ConfigT]):
    def _require(self, *fields: str) -> None:
        missing = [field for field in fields if getattr(self.config, field) is None]
        if missing:
            names = ", ".join(f"`{field}`" for field in missing)
            raise ValueError(f"{type(self).__name__} requires {names}.")

    def _require_one(self, *fields: str) -> None:
        if not any(getattr(self.config, field) is not None for field in fields):
            names = " or ".join(f"`{field}`" for field in fields)
            raise ValueError(f"{type(self).__name__} requires {names}.")

    def _kwargs(self, *fields: str) -> dict[str, Any]:
        return {
            field: getattr(self.config, field)
            for field in fields
            if getattr(self.config, field) is not None
        }


class _PluginConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OAuthProxyConfig(_PluginConfig):
    base_url: AnyHttpUrl | str | None = None
    resource_base_url: AnyHttpUrl | str | None = None
    issuer_url: AnyHttpUrl | str | None = None
    redirect_path: str | None = None
    required_scopes: list[str] | None = None
    allowed_client_redirect_uris: list[str] | None = None
    jwt_signing_key: str | None = None
    require_authorization_consent: ConsentMode = True
    consent_csp_policy: str | None = None
    forward_resource: bool = True


class _OAuthProviderConfig(_OAuthProxyConfig):
    client_id: str | None = None
    client_secret: str | None = None
    timeout_seconds: int = 10
    enable_cimd: bool = True


class _RemoteAuthConfig(_PluginConfig):
    base_url: AnyHttpUrl | str | None = None
    required_scopes: list[str] | None = None
    scopes_supported: list[str] | None = None
    resource_name: str | None = None
    resource_documentation: AnyHttpUrl | None = None


class Auth0AuthConfig(_OAuthProxyConfig):
    """Config model for the Auth0 auth plugin."""

    config_url: AnyHttpUrl | str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    audience: str | None = None


class Auth0Auth(_AuthPlugin[Auth0AuthConfig]):
    """Contribute an `Auth0Provider` as the server's auth provider."""

    meta = PluginMeta(name="auth0-auth")

    def __init__(
        self,
        config: Auth0AuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage

    def auth(self) -> AuthProvider | None:
        self._require("config_url", "client_id", "client_secret", "audience", "base_url")
        return Auth0Provider(
            **self._kwargs(
                "config_url",
                "client_id",
                "client_secret",
                "audience",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "required_scopes",
                "redirect_path",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
            ),
            client_storage=self._client_storage,
        )


class AuthKitAuthConfig(_RemoteAuthConfig):
    """Config model for the WorkOS AuthKit auth plugin."""

    authkit_domain: AnyHttpUrl | str | None = None
    resource_base_url: AnyHttpUrl | str | None = None


class AuthKitAuth(_AuthPlugin[AuthKitAuthConfig]):
    """Contribute an `AuthKitProvider` as the server's auth provider."""

    meta = PluginMeta(name="authkit-auth")

    def __init__(
        self,
        config: AuthKitAuthConfig | dict[str, Any] | None = None,
        *,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        super().__init__(config)
        self._token_verifier = token_verifier

    def auth(self) -> AuthProvider | None:
        self._require("authkit_domain", "base_url")
        return AuthKitProvider(
            **self._kwargs(
                "authkit_domain",
                "base_url",
                "resource_base_url",
                "required_scopes",
                "scopes_supported",
                "resource_name",
                "resource_documentation",
            ),
            token_verifier=self._token_verifier,
        )


class AWSCognitoAuthConfig(_OAuthProxyConfig):
    """Config model for the AWS Cognito auth plugin."""

    user_pool_id: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    aws_region: str = "eu-central-1"
    redirect_path: str | None = "/auth/callback"


class AWSCognitoAuth(_AuthPlugin[AWSCognitoAuthConfig]):
    """Contribute an `AWSCognitoProvider` as the server's auth provider."""

    meta = PluginMeta(name="aws-cognito-auth")

    def __init__(
        self,
        config: AWSCognitoAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage

    def auth(self) -> AuthProvider | None:
        self._require("user_pool_id", "client_id", "client_secret", "base_url")
        return AWSCognitoProvider(
            **self._kwargs(
                "user_pool_id",
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "aws_region",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
            ),
            client_storage=self._client_storage,
        )


class AzureAuthConfig(_OAuthProviderConfig):
    """Config model for the Azure auth plugin."""

    tenant_id: str | None = None
    required_scopes: list[str] | None = None
    identifier_uri: str | None = None
    additional_authorize_scopes: list[str] | None = None
    base_authority: str = "login.microsoftonline.com"


class AzureAuth(_AuthPlugin[AzureAuthConfig]):
    """Contribute an `AzureProvider` as the server's auth provider."""

    meta = PluginMeta(name="azure-auth")

    def __init__(
        self,
        config: AzureAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("client_id", "tenant_id", "required_scopes", "base_url")
        self._require_one("client_secret", "jwt_signing_key")
        return AzureProvider(
            **self._kwargs(
                "client_id",
                "client_secret",
                "tenant_id",
                "required_scopes",
                "base_url",
                "resource_base_url",
                "identifier_uri",
                "issuer_url",
                "redirect_path",
                "additional_authorize_scopes",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "base_authority",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )


class ClerkAuthConfig(_OAuthProviderConfig):
    """Config model for the Clerk auth plugin."""

    domain: str | None = None
    valid_scopes: list[str] | None = None
    extra_authorize_params: dict[str, str] | None = None


class ClerkAuth(_AuthPlugin[ClerkAuthConfig]):
    """Contribute a `ClerkProvider` as the server's auth provider."""

    meta = PluginMeta(name="clerk-auth")

    def __init__(
        self,
        config: ClerkAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("domain", "client_id", "base_url")
        self._require_one("client_secret", "jwt_signing_key")
        return ClerkProvider(
            **self._kwargs(
                "domain",
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "valid_scopes",
                "timeout_seconds",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "extra_authorize_params",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )


class DescopeAuthConfig(_RemoteAuthConfig):
    """Config model for the Descope auth plugin."""

    config_url: AnyHttpUrl | str | None = None
    project_id: str | None = None
    descope_base_url: AnyHttpUrl | str | None = None


class DescopeAuth(_AuthPlugin[DescopeAuthConfig]):
    """Contribute a `DescopeProvider` as the server's auth provider."""

    meta = PluginMeta(name="descope-auth")

    def __init__(
        self,
        config: DescopeAuthConfig | dict[str, Any] | None = None,
        *,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        super().__init__(config)
        self._token_verifier = token_verifier

    def auth(self) -> AuthProvider | None:
        self._require("base_url")
        if self.config.config_url is None:
            self._require("project_id", "descope_base_url")
        return DescopeProvider(
            **self._kwargs(
                "base_url",
                "config_url",
                "project_id",
                "descope_base_url",
                "required_scopes",
                "scopes_supported",
                "resource_name",
                "resource_documentation",
            ),
            token_verifier=self._token_verifier,
        )


class DiscordAuthConfig(_OAuthProviderConfig):
    """Config model for the Discord auth plugin."""


class DiscordAuth(_AuthPlugin[DiscordAuthConfig]):
    """Contribute a `DiscordProvider` as the server's auth provider."""

    meta = PluginMeta(name="discord-auth")

    def __init__(
        self,
        config: DiscordAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("client_id", "client_secret", "base_url")
        return DiscordProvider(
            **self._kwargs(
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "timeout_seconds",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )


class GitHubAuthConfig(_OAuthProviderConfig):
    """Config model for the GitHub auth plugin."""

    cache_ttl_seconds: int | None = None
    max_cache_size: int | None = None


class GitHubAuth(_AuthPlugin[GitHubAuthConfig]):
    """Contribute a `GitHubProvider` as the server's auth provider."""

    meta = PluginMeta(name="github-auth")

    def __init__(
        self,
        config: GitHubAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("client_id", "client_secret", "base_url")
        return GitHubProvider(
            **self._kwargs(
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "timeout_seconds",
                "cache_ttl_seconds",
                "max_cache_size",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )


class GoogleAuthConfig(_OAuthProviderConfig):
    """Config model for the Google auth plugin."""

    valid_scopes: list[str] | None = None
    extra_authorize_params: dict[str, str] | None = None


class GoogleAuth(_AuthPlugin[GoogleAuthConfig]):
    """Contribute a `GoogleProvider` as the server's auth provider."""

    meta = PluginMeta(name="google-auth")

    def __init__(
        self,
        config: GoogleAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("client_id", "base_url")
        self._require_one("client_secret", "jwt_signing_key")
        return GoogleProvider(
            **self._kwargs(
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "valid_scopes",
                "timeout_seconds",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "extra_authorize_params",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )


class KeycloakAuthConfig(_PluginConfig):
    """Config model for the Keycloak auth plugin."""

    realm_url: AnyHttpUrl | str | None = None
    base_url: AnyHttpUrl | str | None = None
    required_scopes: list[str] | str | None = None
    audience: str | list[str] | None = None


class KeycloakAuth(_AuthPlugin[KeycloakAuthConfig]):
    """Contribute a `KeycloakAuthProvider` as the server's auth provider."""

    meta = PluginMeta(name="keycloak-auth")

    def __init__(
        self,
        config: KeycloakAuthConfig | dict[str, Any] | None = None,
        *,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        super().__init__(config)
        self._token_verifier = token_verifier

    def auth(self) -> AuthProvider | None:
        self._require("realm_url", "base_url")
        return KeycloakAuthProvider(
            **self._kwargs("realm_url", "base_url", "required_scopes", "audience"),
            token_verifier=self._token_verifier,
        )


class OCIAuthConfig(_OAuthProxyConfig):
    """Config model for the OCI auth plugin."""

    config_url: AnyHttpUrl | str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    audience: str | None = None


class OCIAuth(_AuthPlugin[OCIAuthConfig]):
    """Contribute an `OCIProvider` as the server's auth provider."""

    meta = PluginMeta(name="oci-auth")

    def __init__(
        self,
        config: OCIAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage

    def auth(self) -> AuthProvider | None:
        self._require("config_url", "client_id", "client_secret", "base_url")
        return OCIProvider(
            **self._kwargs(
                "config_url",
                "client_id",
                "client_secret",
                "base_url",
                "resource_base_url",
                "audience",
                "issuer_url",
                "required_scopes",
                "redirect_path",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
            ),
            client_storage=self._client_storage,
        )


class PropelAuthConfig(_RemoteAuthConfig):
    """Config model for the PropelAuth auth plugin."""

    auth_url: AnyHttpUrl | str | None = None
    introspection_client_id: str | None = None
    introspection_client_secret: str | None = None
    resource: AnyHttpUrl | str | None = None
    introspection_timeout_seconds: int | None = None
    introspection_cache_ttl_seconds: int | None = None
    introspection_max_cache_size: int | None = None


class PropelAuth(_AuthPlugin[PropelAuthConfig]):
    """Contribute a `PropelAuthProvider` as the server's auth provider."""

    meta = PluginMeta(name="propelauth-auth")

    def __init__(
        self,
        config: PropelAuthConfig | dict[str, Any] | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require(
            "auth_url",
            "introspection_client_id",
            "introspection_client_secret",
            "base_url",
        )
        overrides: PropelAuthTokenIntrospectionOverrides = {}
        if self.config.introspection_timeout_seconds is not None:
            overrides["timeout_seconds"] = self.config.introspection_timeout_seconds
        if self.config.introspection_cache_ttl_seconds is not None:
            overrides["cache_ttl_seconds"] = self.config.introspection_cache_ttl_seconds
        if self.config.introspection_max_cache_size is not None:
            overrides["max_cache_size"] = self.config.introspection_max_cache_size
        if self._http_client is not None:
            overrides["http_client"] = self._http_client

        return PropelAuthProvider(
            **self._kwargs(
                "auth_url",
                "introspection_client_id",
                "introspection_client_secret",
                "base_url",
                "required_scopes",
                "scopes_supported",
                "resource_name",
                "resource_documentation",
                "resource",
            ),
            token_introspection_overrides=overrides or None,
        )


class ScalekitAuthConfig(_RemoteAuthConfig):
    """Config model for the Scalekit auth plugin."""

    environment_url: AnyHttpUrl | str | None = None
    resource_id: str | None = None
    mcp_url: AnyHttpUrl | str | None = None
    client_id: str | None = None


class ScalekitAuth(_AuthPlugin[ScalekitAuthConfig]):
    """Contribute a `ScalekitProvider` as the server's auth provider."""

    meta = PluginMeta(name="scalekit-auth")

    def __init__(
        self,
        config: ScalekitAuthConfig | dict[str, Any] | None = None,
        *,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        super().__init__(config)
        self._token_verifier = token_verifier

    def auth(self) -> AuthProvider | None:
        self._require("environment_url", "resource_id")
        self._require_one("base_url", "mcp_url")
        return ScalekitProvider(
            **self._kwargs(
                "environment_url",
                "resource_id",
                "base_url",
                "mcp_url",
                "client_id",
                "required_scopes",
                "scopes_supported",
                "resource_name",
                "resource_documentation",
            ),
            token_verifier=self._token_verifier,
        )


class SupabaseAuthConfig(_RemoteAuthConfig):
    """Config model for the Supabase auth plugin."""

    project_url: AnyHttpUrl | str | None = None
    auth_route: str = "/auth/v1"
    algorithm: Algorithm = "ES256"


class SupabaseAuth(_AuthPlugin[SupabaseAuthConfig]):
    """Contribute a `SupabaseProvider` as the server's auth provider."""

    meta = PluginMeta(name="supabase-auth")

    def __init__(
        self,
        config: SupabaseAuthConfig | dict[str, Any] | None = None,
        *,
        token_verifier: TokenVerifier | None = None,
    ) -> None:
        super().__init__(config)
        self._token_verifier = token_verifier

    def auth(self) -> AuthProvider | None:
        self._require("project_url", "base_url")
        return SupabaseProvider(
            **self._kwargs(
                "project_url",
                "base_url",
                "auth_route",
                "algorithm",
                "required_scopes",
                "scopes_supported",
                "resource_name",
                "resource_documentation",
            ),
            token_verifier=self._token_verifier,
        )


class WorkOSAuthConfig(_OAuthProviderConfig):
    """Config model for the WorkOS auth plugin."""

    authkit_domain: str | None = None


class WorkOSAuth(_AuthPlugin[WorkOSAuthConfig]):
    """Contribute a `WorkOSProvider` as the server's auth provider."""

    meta = PluginMeta(name="workos-auth")

    def __init__(
        self,
        config: WorkOSAuthConfig | dict[str, Any] | None = None,
        *,
        client_storage: AsyncKeyValue | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._client_storage = client_storage
        self._http_client = http_client

    def auth(self) -> AuthProvider | None:
        self._require("client_id", "client_secret", "authkit_domain", "base_url")
        return WorkOSProvider(
            **self._kwargs(
                "client_id",
                "client_secret",
                "authkit_domain",
                "base_url",
                "resource_base_url",
                "issuer_url",
                "redirect_path",
                "required_scopes",
                "timeout_seconds",
                "allowed_client_redirect_uris",
                "jwt_signing_key",
                "require_authorization_consent",
                "consent_csp_policy",
                "forward_resource",
                "enable_cimd",
            ),
            client_storage=self._client_storage,
            http_client=self._http_client,
        )
