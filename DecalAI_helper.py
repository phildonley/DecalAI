# DecalAI_helper.py

import os
import json
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Name of the file that will hold the saved API key (created next to this helper)
API_KEY_FILENAME = "api_key.json"


def _ask_user_for_api_key():
    """
    Pops up a simple Tkinter dialog asking for the API key.
    Returns the string the user entered (or None if they canceled).
    """
    root = tk.Tk()
    root.withdraw()  # hide the main window

    api_key = simpledialog.askstring(
        "API Key Required",
        (
            "Please enter your API key for the Digital Library service:\n\n"
            "(If you don't have one, request it from your IT administrator.)"
        ),
        parent=root
    )
    root.destroy()
    return api_key.strip() if api_key else None


def _load_api_key_from_file(config_path):
    """
    Attempts to load {"api_key": "..."} from config_path.
    If successful and non-empty, returns the key string;
    otherwise returns None.
    """
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("api_key", "").strip()
        return key if key else None
    except Exception:
        # Any error reading/parsing → treat as missing
        return None


def _save_api_key_to_file(config_path, api_key):
    """
    Writes {"api_key": "<the_key>"} to config_path (overwriting if necessary).
    If writing fails, shows an error and exits.
    """
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"api_key": api_key}, f)
    except Exception as e:
        messagebox.showerror(
            "Error Saving API Key",
            (
                f"Could not write to {config_path}:\n\n{e}\n\n"
                "Please ensure you have write permission in this directory."
            )
        )
        sys.exit(1)


def get_valid_api_key():
    """
    Returns a valid API key string. Steps:
      1) Look in <helper_folder>/api_key.json for a saved key.
      2) If not found or empty, prompt the user via a Tkinter dialog.
      3) Save whatever the user types into api_key.json for future runs.
      4) Return the non-empty key to the caller.

    If the user cancels or enters nothing, shows a warning and re-prompts.
    If they choose not to try again, exits cleanly.
    """
    # Determine directory where this helper file lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, API_KEY_FILENAME)

    # 1) Attempt to load an existing key from disk
    key = _load_api_key_from_file(config_path)
    if key:
        return key

    # 2) Not found or empty → prompt the user
    while True:
        user_key = _ask_user_for_api_key()
        if user_key:
            # Save the new key for next time
            _save_api_key_to_file(config_path, user_key)
            return user_key
        else:
            # If user canceled or submitted empty, ask whether to retry
            retry = messagebox.askyesno(
                "API Key Required",
                (
                    "No API key was provided. The application cannot continue without a valid API key.\n\n"
                    "Would you like to try again?\n"
                    "(Yes = show the prompt again; No = exit)"
                )
            )
            if not retry:
                sys.exit(0)
