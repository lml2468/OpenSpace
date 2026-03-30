"""OpenClaw host-agent config reader.

Reads ``~/.openclaw/openclaw.json`` to auto-detect:
  - LLM provider credentials (via ``auth-profiles`` — not yet implemented)
  - Skill-level env block (``skills.entries.openspace.env``)
  - OpenAI API key for embedding generation

Config path resolution mirrors OpenClaw's own logic:
  1. ``OPENCLAW_CONFIG_PATH`` env var
  2. ``OPENCLAW_STATE_DIR/openclaw.json``
  3. ``~/.openclaw/openclaw.json`` (default)

Fallback legacy dirs: ``~/.clawdbot``, ``~/.moldbot``, ``~/.moltbot``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("openspace.host_detection")

_STATE_DIRNAMES = [".openclaw", ".clawdbot", ".moldbot", ".moltbot"]
_CONFIG_FILENAMES = ["openclaw.json", "clawdbot.json", "moldbot.json", "moltbot.json"]


def _resolve_openclaw_config_path() -> Optional[Path]:
    """Find the OpenClaw config file on disk."""
    import os

    # 1. Explicit env override
    explicit = os.environ.get("OPENCLAW_CONFIG_PATH", "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        return None

    # 2. State dir override
    state_dir = os.environ.get("OPENCLAW_STATE_DIR", "").strip()
    if state_dir:
        for fname in _CONFIG_FILENAMES:
            p = Path(state_dir) / fname
            if p.is_file():
                return p

    # 3. Default locations
    home = Path.home()
    for dirname in _STATE_DIRNAMES:
        for fname in _CONFIG_FILENAMES:
            p = home / dirname / fname
            if p.is_file():
                return p

    return None


def _load_openclaw_config() -> Optional[Dict[str, Any]]:
    """Load and parse the OpenClaw config file.  Returns None on failure."""
    config_path = _resolve_openclaw_config_path()
    if config_path is None:
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read OpenClaw config %s: %s", config_path, e)
        return None


def read_openclaw_skill_env(skill_name: str = "openspace") -> Dict[str, str]:
    """Read ``skills.entries.<skill_name>.env`` from OpenClaw config.

    This is the OpenClaw equivalent of nanobot's
    ``tools.mcpServers.openspace.env``.

    Returns the env dict (empty if not found / parse error).
    """
    data = _load_openclaw_config()
    if data is None:
        return {}

    skills = data.get("skills", {})
    if not isinstance(skills, dict):
        return {}
    entries = skills.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    skill_cfg = entries.get(skill_name, {})
    if not isinstance(skill_cfg, dict):
        return {}
    env_block = skill_cfg.get("env", {})
    return env_block if isinstance(env_block, dict) else {}


def get_openclaw_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from OpenClaw config.

    Checks ``skills.entries.openspace.env.OPENAI_API_KEY`` first,
    then any top-level env vars in the config.

    Returns the key string, or None.
    """
    # Try skill-level env
    env = read_openclaw_skill_env("openspace")
    key = env.get("OPENAI_API_KEY", "").strip()
    if key:
        logger.debug("Using OpenAI API key from OpenClaw skill env config")
        return key

    # Try top-level config env.vars
    data = _load_openclaw_config()
    if data:
        env_section = data.get("env", {})
        if isinstance(env_section, dict):
            vars_block = env_section.get("vars", {})
            if isinstance(vars_block, dict):
                key = vars_block.get("OPENAI_API_KEY", "").strip()
                if key:
                    logger.debug("Using OpenAI API key from OpenClaw env.vars config")
                    return key

    return None


def is_openclaw_host() -> bool:
    """Detect if the current environment is running under OpenClaw."""
    import os
    # Check OpenClaw-specific env vars
    if os.environ.get("OPENCLAW_STATE_DIR") or os.environ.get("OPENCLAW_CONFIG_PATH"):
        return True
    # Check if config exists
    return _resolve_openclaw_config_path() is not None


def try_read_openclaw_config(model: str) -> Optional[Dict[str, Any]]:
    """Read LLM credentials from ``~/.openclaw/openclaw.json``.

    OpenClaw stores providers under ``models.providers.<name>`` with fields:
      - ``apiKey``   → litellm ``api_key``
      - ``baseUrl``  → litellm ``api_base``
      - ``api``      → determines litellm model prefix
                        (``anthropic-messages`` → ``anthropic/``, etc.)
      - ``models[]`` → available model list

    Default model is resolved from ``agents.defaults.model.primary``
    (format: ``provider/model-id``, e.g. ``mlamp/claude-opus-4-6``).

    Resolution:
      1. If *model* is given, extract provider prefix and look it up.
      2. Otherwise use the default model from agents config.
      3. Match provider → extract credentials.

    Returns litellm kwargs dict, or None.  May include ``_model`` and
    ``_forced_provider`` keys for downstream use.
    """
    data = _load_openclaw_config()
    if data is None:
        return None

    models_section = data.get("models", {})
    if not isinstance(models_section, dict):
        return None
    providers = models_section.get("providers", {})
    if not isinstance(providers, dict) or not providers:
        return None

    # --- Resolve default model from agents config ---
    agents = data.get("agents", {})
    default_model = ""
    if isinstance(agents, dict):
        defaults = agents.get("defaults", {})
        if isinstance(defaults, dict):
            model_cfg = defaults.get("model", {})
            if isinstance(model_cfg, dict):
                default_model = model_cfg.get("primary", "")
            elif isinstance(model_cfg, str):
                default_model = model_cfg

    # --- Determine which provider to use ---
    target_model = model or default_model or ""
    provider_name = ""
    model_id = ""

    if "/" in target_model:
        # Format: provider/model-id (e.g. mlamp/claude-opus-4-6)
        provider_name, model_id = target_model.split("/", 1)
    elif target_model and default_model and "/" in default_model:
        # Model given without prefix, use default's provider
        provider_name = default_model.split("/", 1)[0]
        model_id = target_model

    if not provider_name:
        # Fallback: first provider with an apiKey
        for name, prov in providers.items():
            if isinstance(prov, dict) and prov.get("apiKey"):
                provider_name = name
                break

    if not provider_name:
        return None

    prov_config = providers.get(provider_name)
    if not isinstance(prov_config, dict):
        return None

    api_key = prov_config.get("apiKey", "")
    if not api_key:
        return None

    result: Dict[str, Any] = {"api_key": api_key}

    base_url = prov_config.get("baseUrl", "")
    if base_url:
        # Ensure the base URL ends with /v1 for OpenAI-compatible proxies.
        # Without it, litellm hits the proxy's dashboard page instead of
        # the API endpoint (e.g. New API / One API gateways).
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        result["api_base"] = base_url

    # --- Resolve model name ---
    # OpenClaw providers typically expose models behind an OpenAI-compatible
    # proxy (baseUrl + /v1/chat/completions).  We must prefix the model
    # with ``openai/`` so that litellm uses the OpenAI request format
    # instead of guessing from the model name (e.g. "claude" → Anthropic
    # SDK, which would hit the wrong endpoint on the proxy and 404).
    resolved_model_id = model_id
    if not resolved_model_id and default_model and "/" in default_model:
        resolved_model_id = default_model.split("/", 1)[1]

    if resolved_model_id:
        result["_model"] = f"openai/{resolved_model_id}"

    logger.info(
        "Auto-detected LLM credentials from OpenClaw config (%s), "
        "provider=%r, model=%r",
        _resolve_openclaw_config_path(),
        provider_name,
        result.get("_model", ""),
    )

    return result

