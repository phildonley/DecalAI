# ── DecalAI_helper.py ─────────────────────────────────────────────────────────
import os
import json
import tkinter as tk
from tkinter import simpledialog

def get_valid_api_key():
    """
    If there’s already an “api_key.json” sitting next to this helper, read it and
    return the stored key. Otherwise, pop up a small Tkinter dialog box to let the
    user paste/type their key, save it to “api_key.json”, and return it.
    
    Raises:
        Exception: if the user cancels or does not enter anything.
    """
    # 1) Define where we’ll store the key:
    helper_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(helper_dir, "api_key.json")

    # 2) Try to load an existing key if present:
    api_key = None
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                data = json.load(f)
                api_key = data.get("api_key")
        except Exception:
            # If the file is malformed, just ignore it and re-prompt
            api_key = None

    # 3) If no key was found (or was invalid), pop up a dialog to ask for one:
    if not api_key:
        root = tk.Tk()
        root.withdraw()   # hide the main root window

        prompt = (
            "No API key was found.  \n"
            "Please paste your “x-api-key” here (you can copy/paste from your email or IT ticket):"
        )
        # simpledialog.askstring returns None if the user cancels
        api_key = simpledialog.askstring(title="DecalAI: Enter API Key", prompt=prompt)

        root.destroy()

        if not api_key or not api_key.strip():
            raise Exception(
                "DecalAI requires a valid API key to proceed.  "
                "Please run again and enter your key when prompted."
            )

        # 4) Save the key to disk so next time we skip the prompt:
        try:
            with open(cfg_path, "w") as f:
                json.dump({"api_key": api_key.strip()}, f, indent=2)
        except Exception as e:
            # If we can’t write the file, warn but still return the key
            print(f"⚠️ Warning: Failed to save API key to {cfg_path}: {e}")

    return api_key.strip()


# If somebody runs this helper directly, we can test that it at least returns a string.
if __name__ == "__main__":
    try:
        k = get_valid_api_key()
        print(f"✅ Got API key: {k[:4]}… (length={len(k)})")
    except Exception as e:
        print(f"❌ {e}")