import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path


def install_plugin(url: str, target_dir: str = "plugins"):
    """Clone and install a plugin from a git URL."""
    # Derive directory name from URL
    repo_name = url.split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    plugins_root = Path(target_dir)
    plugins_root.mkdir(exist_ok=True)

    plugin_path = plugins_root / repo_name

    if plugin_path.exists():
        print(f"Directory {plugin_path} already exists. Skipping clone.")
    else:
        print(f"Cloning {url} into {plugin_path}...")
        try:
            subprocess.check_call(["git", "clone", url, str(plugin_path)])
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            sys.exit(1)

    print(f"Installing {repo_name} in editable mode...")
    try:
        # Check if we are in a poetry project
        use_poetry = False
        if Path("pyproject.toml").exists():
            try:
                subprocess.check_call(
                    ["poetry", "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                use_poetry = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        if use_poetry:
            print("Detected Poetry project. Installing via 'poetry add'...")
            subprocess.check_call(
                ["poetry", "add", "--editable", f"./{plugin_path}"]
            )
        else:
            # Use the current executable to ensure we install into the same environment
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", str(plugin_path)]
            )
        print(f"Successfully installed {repo_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing plugin: {e}")
        sys.exit(1)


def list_plugins():
    """List installed plugins registered via entry points."""
    print("Installed biolm plugins:")
    # For Python 3.10+
    if hasattr(importlib.metadata, "entry_points"):
        eps = importlib.metadata.entry_points(group="biolm.plugins")
    else:
        eps = importlib.metadata.entry_points().get("biolm.plugins", [])

    if not eps:
        print("  (none)")
        return

    for ep in eps:
        dist = ep.dist
        location = "unknown"
        version = "unknown"
        if dist:
            version = dist.version
            # Try to find location for editable installs
            # direct_url.json is often present for pip installs
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                import json

                try:
                    data = json.loads(direct_url)
                    if "url" in data:
                        location = data["url"]
                except:
                    pass

        print(f"  - {ep.name} (version: {version})")
        # print(f"    Location: {location}")


def remove_plugin(plugin_name: str):
    """Uninstall a plugin."""
    # We need to find the package name associated with the plugin name
    # This is tricky because the entry point name might not match the package name.
    # But usually they are related.

    # Let's search entry points to find the distribution
    target_dist = None
    if hasattr(importlib.metadata, "entry_points"):
        eps = importlib.metadata.entry_points(group="biolm.plugins")
    else:
        eps = importlib.metadata.entry_points().get("biolm.plugins", [])

    for ep in eps:
        if ep.name == plugin_name:
            target_dist = ep.dist
            break

    if not target_dist:
        print(f"Plugin '{plugin_name}' not found.")
        return

    package_name = target_dist.metadata["Name"]
    print(f"Found plugin '{plugin_name}' in package '{package_name}'.")

    confirm = input(f"Are you sure you want to uninstall '{package_name}'? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    print(f"Uninstalling {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package_name])
        print(f"Successfully uninstalled {package_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error uninstalling plugin: {e}")
        sys.exit(1)


def handle_plugin_command(args):
    parser = argparse.ArgumentParser(
        prog="biolm plugin", description="Manage biolm plugins"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Install command
    install_parser = subparsers.add_parser(
        "install", help="Install a plugin from a git URL"
    )
    install_parser.add_argument("url", help="Git URL of the plugin repository")
    install_parser.add_argument(
        "--dir", default="plugins", help="Directory to clone into (default: plugins)"
    )

    # List command
    subparsers.add_parser("list", help="List installed plugins")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove (uninstall) a plugin")
    remove_parser.add_argument("name", help="Name of the plugin to remove")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "install":
        install_plugin(parsed_args.url, parsed_args.dir)
    elif parsed_args.command == "list":
        list_plugins()
    elif parsed_args.command == "remove":
        remove_plugin(parsed_args.name)
    else:
        parser.print_help()
