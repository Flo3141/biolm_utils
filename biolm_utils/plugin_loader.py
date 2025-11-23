"""Plugin discovery helpers for biolm_utils.

Provide a small, testable discovery mechanism that can be used to discover
installed plugins via Python entry-points and as a fallback import from a
`plugins/` directory on the current PYTHONPATH.

The discovery intentionally only *registers* plugins (by calling the
plugin_registry.register_plugin) — it does not alter runtime behaviour other
than making the plugin factories available through the registry.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Iterable, Optional

from .plugin_registry import get_plugin_factory, register_plugin


def discover_entrypoint_plugins(group: str = "biolm_utils.plugins") -> list[str]:
    """Discover and register plugins defined by distribution entry-points.

    This inspects importlib.metadata.entry_points() for the given group and
    will attempt to load and register each entry.

    Returns a list of plugin names successfully registered.
    """
    registered = []
    eps = entry_points()
    # importlib.metadata.EntryPoints object is queryable via .select in py3.10+
    try:
        group_eps = eps.select(group=group)
    except Exception:
        # Fallback for older importlib.metadata APIs
        group_eps = [ep for ep in eps if getattr(ep, "group", None) == group]

    for ep in group_eps:
        name = ep.name
        # Skip if already registered
        if get_plugin_factory(name) is not None:
            continue

        try:
            factory = ep.load()
        except Exception:
            # Loading failed; skip plugin but continue discovering others.
            continue

        # Register the entry-point factory under its name
        register_plugin(name, factory)
        registered.append(name)

    return registered


def discover_plugins_from_dir(plugins_dir: Optional[Path | str] = None) -> list[str]:
    """Discover and register plugins by importing modules from a plugins dir.

    The function accepts a path (default: './plugins') and imports every
    top-level module found. If the module exposes a `register_plugin` callable
    or a `get_{name}_config` factory the loader will call it and register
    accordingly.

    Returns a list of plugin names that were registered.
    """
    registered = []
    path = Path(plugins_dir or "plugins")
    if not path.exists() or not path.is_dir():
        return registered

    # Treat each child that looks like a package or python module as a candidate
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue

        module_name = None
        if child.is_file() and child.suffix == ".py":
            module_name = child.stem
        elif child.is_dir() and (child / "__init__.py").exists():
            module_name = child.name

        if module_name is None:
            continue

        try:
            mod = import_module(module_name)
        except Exception:
            # import errors are ignored — plugin authors should fix their modules
            continue

        # Prefer explicit register_plugin function if present
        if hasattr(mod, "register_plugin"):
            try:
                mod.register_plugin()
                registered.append(module_name)
            except Exception:
                continue
            continue

        # Look for a factory function named get_<module>_config or get_config
        factory = None
        factory_name = f"get_{module_name}_config"
        if hasattr(mod, factory_name):
            factory = getattr(mod, factory_name)
        elif hasattr(mod, "get_config"):
            factory = getattr(mod, "get_config")

        if callable(factory):
            # register under module name unless factory declares a different name
            register_plugin(module_name, factory)
            registered.append(module_name)

    return registered


def discover_all_plugins() -> list[str]:
    """Discover plugins with both entry-point and plugins/ dir strategies.

    The method will first attempt to register entry-point plugins and then
    attempt the plugins directory fallback.
    Returns the combined list of newly registered plugin names.
    """
    names = discover_entrypoint_plugins()
    names += [n for n in discover_plugins_from_dir() if n not in names]
    return names
