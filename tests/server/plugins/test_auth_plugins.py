"""Tests for first-party auth plugin wrappers."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from fastmcp import FastMCP
from fastmcp.server.auth.oidc_proxy import OIDCConfiguration
from fastmcp.server.auth.providers.auth0 import Auth0Provider
from fastmcp.server.auth.providers.aws import AWSCognitoProvider
from fastmcp.server.auth.providers.azure import AzureProvider
from fastmcp.server.auth.providers.clerk import ClerkProvider
from fastmcp.server.auth.providers.descope import DescopeProvider
from fastmcp.server.auth.providers.discord import DiscordProvider
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.auth.providers.google import GoogleProvider
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.auth.providers.keycloak import KeycloakAuthProvider
from fastmcp.server.auth.providers.oci import OCIProvider
from fastmcp.server.auth.providers.propelauth import PropelAuthProvider
from fastmcp.server.auth.providers.scalekit import ScalekitProvider
from fastmcp.server.auth.providers.supabase import SupabaseProvider
from fastmcp.server.auth.providers.workos import AuthKitProvider, WorkOSProvider
from fastmcp.server.plugins.auth import (
    Auth0Auth,
    Auth0AuthConfig,
    AuthKitAuth,
    AuthKitAuthConfig,
    AWSCognitoAuth,
    AWSCognitoAuthConfig,
    AzureAuth,
    AzureAuthConfig,
    ClerkAuth,
    ClerkAuthConfig,
    DescopeAuth,
    DescopeAuthConfig,
    DiscordAuth,
    DiscordAuthConfig,
    GitHubAuth,
    GitHubAuthConfig,
    GoogleAuth,
    GoogleAuthConfig,
    KeycloakAuth,
    KeycloakAuthConfig,
    OCIAuth,
    OCIAuthConfig,
    PropelAuth,
    PropelAuthConfig,
    ScalekitAuth,
    ScalekitAuthConfig,
    SupabaseAuth,
    SupabaseAuthConfig,
    WorkOSAuth,
    WorkOSAuthConfig,
)


def _verifier() -> StaticTokenVerifier:
    return StaticTokenVerifier(tokens={"t": {"client_id": "c", "scopes": []}})


def _oidc_config() -> OIDCConfiguration:
    return OIDCConfiguration.model_validate(
        {
            "issuer": "https://idp.example.com",
            "authorization_endpoint": "https://idp.example.com/authorize",
            "token_endpoint": "https://idp.example.com/token",
            "jwks_uri": "https://idp.example.com/jwks.json",
            "response_types_supported": ["code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
        }
    )


@pytest.fixture(autouse=True)
def _mock_oidc_discovery():
    with patch(
        "fastmcp.server.auth.oidc_proxy.OIDCConfiguration.get_oidc_configuration",
        return_value=_oidc_config(),
    ):
        yield


PROVIDER_CASES: list[tuple[type, type, dict[str, Any], type]] = [
    (
        Auth0Auth,
        Auth0AuthConfig,
        {
            "config_url": "https://idp.example.com/.well-known/openid-configuration",
            "client_id": "client",
            "client_secret": "secret",
            "audience": "audience",
            "base_url": "https://mcp.example.com",
        },
        Auth0Provider,
    ),
    (
        AuthKitAuth,
        AuthKitAuthConfig,
        {
            "authkit_domain": "https://example.authkit.app",
            "base_url": "https://mcp.example.com",
        },
        AuthKitProvider,
    ),
    (
        AWSCognitoAuth,
        AWSCognitoAuthConfig,
        {
            "user_pool_id": "us-east-1_abc",
            "client_id": "client",
            "client_secret": "secret",
            "aws_region": "us-east-1",
            "base_url": "https://mcp.example.com",
        },
        AWSCognitoProvider,
    ),
    (
        AzureAuth,
        AzureAuthConfig,
        {
            "client_id": "client",
            "client_secret": "secret",
            "tenant_id": "tenant",
            "required_scopes": ["read"],
            "base_url": "https://mcp.example.com",
        },
        AzureProvider,
    ),
    (
        ClerkAuth,
        ClerkAuthConfig,
        {
            "domain": "example.clerk.accounts.dev",
            "client_id": "client",
            "client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        ClerkProvider,
    ),
    (
        DescopeAuth,
        DescopeAuthConfig,
        {
            "config_url": "https://api.descope.com/v1/apps/agentic/P123/M456/.well-known/openid-configuration",
            "base_url": "https://mcp.example.com",
        },
        DescopeProvider,
    ),
    (
        DiscordAuth,
        DiscordAuthConfig,
        {
            "client_id": "client",
            "client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        DiscordProvider,
    ),
    (
        GitHubAuth,
        GitHubAuthConfig,
        {
            "client_id": "client",
            "client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        GitHubProvider,
    ),
    (
        GoogleAuth,
        GoogleAuthConfig,
        {
            "client_id": "client",
            "client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        GoogleProvider,
    ),
    (
        KeycloakAuth,
        KeycloakAuthConfig,
        {
            "realm_url": "https://keycloak.example.com/realms/main",
            "base_url": "https://mcp.example.com",
        },
        KeycloakAuthProvider,
    ),
    (
        OCIAuth,
        OCIAuthConfig,
        {
            "config_url": "https://idp.example.com/.well-known/openid-configuration",
            "client_id": "client",
            "client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        OCIProvider,
    ),
    (
        PropelAuth,
        PropelAuthConfig,
        {
            "auth_url": "https://auth.example.com",
            "introspection_client_id": "client",
            "introspection_client_secret": "secret",
            "base_url": "https://mcp.example.com",
        },
        PropelAuthProvider,
    ),
    (
        ScalekitAuth,
        ScalekitAuthConfig,
        {
            "environment_url": "https://env.scalekit.com",
            "resource_id": "res_123",
            "base_url": "https://mcp.example.com",
        },
        ScalekitProvider,
    ),
    (
        SupabaseAuth,
        SupabaseAuthConfig,
        {
            "project_url": "https://abc123.supabase.co",
            "base_url": "https://mcp.example.com",
        },
        SupabaseProvider,
    ),
    (
        WorkOSAuth,
        WorkOSAuthConfig,
        {
            "client_id": "client",
            "client_secret": "secret",
            "authkit_domain": "https://example.authkit.app",
            "base_url": "https://mcp.example.com",
        },
        WorkOSProvider,
    ),
]


def _plugin_kwargs(plugin_cls: type) -> dict[str, Any]:
    if plugin_cls in {
        AuthKitAuth,
        DescopeAuth,
        KeycloakAuth,
        ScalekitAuth,
        SupabaseAuth,
    }:
        return {"token_verifier": _verifier()}
    return {}


class TestAuthProviderPlugins:
    @pytest.mark.parametrize(
        ("plugin_cls", "config_cls", "config", "provider_cls"), PROVIDER_CASES
    )
    def test_config_generic_binding(self, plugin_cls, config_cls, config, provider_cls):
        assert plugin_cls._config_cls is config_cls

    @pytest.mark.parametrize(
        ("plugin_cls", "config_cls", "config", "provider_cls"), PROVIDER_CASES
    )
    def test_default_config_instantiable(
        self, plugin_cls, config_cls, config, provider_cls
    ):
        assert config_cls()

    @pytest.mark.parametrize(
        ("plugin_cls", "config_cls", "config", "provider_cls"), PROVIDER_CASES
    )
    def test_unknown_config_key_rejected(
        self, plugin_cls, config_cls, config, provider_cls
    ):
        with pytest.raises((ValidationError, Exception), match="forbid|extra"):
            config_cls(not_a_real_option=True)

    @pytest.mark.parametrize(
        ("plugin_cls", "config_cls", "config", "provider_cls"), PROVIDER_CASES
    )
    def test_auth_builds_provider(self, plugin_cls, config_cls, config, provider_cls):
        auth = plugin_cls(config, **_plugin_kwargs(plugin_cls)).auth()

        assert isinstance(auth, provider_cls)

    @pytest.mark.parametrize(
        ("plugin_cls", "config_cls", "config", "provider_cls"), PROVIDER_CASES
    )
    def test_plugin_installs_as_server_auth(
        self, plugin_cls, config_cls, config, provider_cls
    ):
        plugin = plugin_cls(config, **_plugin_kwargs(plugin_cls))

        mcp = FastMCP("t", plugins=[plugin])

        assert isinstance(mcp.auth, provider_cls)

    @pytest.mark.parametrize("missing", ["project_url", "base_url"])
    def test_required_fields_checked_when_auth_builds(self, missing: str):
        config = {
            "project_url": "https://abc123.supabase.co",
            "base_url": "https://mcp.example.com",
        }
        del config[missing]

        plugin = SupabaseAuth(config, token_verifier=_verifier())

        with pytest.raises(ValueError, match=missing):
            plugin.auth()

    def test_supabase_passthroughs_config_and_python_verifier(self):
        verifier = _verifier()
        plugin = SupabaseAuth(
            SupabaseAuthConfig(
                project_url="https://abc123.supabase.co",
                base_url="https://mcp.example.com",
                required_scopes=["read"],
                scopes_supported=["read", "write"],
                resource_name="Example MCP",
            ),
            token_verifier=verifier,
        )

        auth = plugin.auth()

        assert isinstance(auth, SupabaseProvider)
        assert auth.token_verifier is verifier
        assert str(auth.base_url).rstrip("/") == "https://mcp.example.com"
        assert auth._scopes_supported == ["read", "write"]
        assert auth.resource_name == "Example MCP"
