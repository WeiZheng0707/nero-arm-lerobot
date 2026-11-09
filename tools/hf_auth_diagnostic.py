"""Check HF authentication; optionally save login with --save-login."""
import argparse
import getpass
import os
import re
import sys

import requests


def check(session, label, url, token=None, method="GET"):
    headers = {"Authorization": "Bearer " + token} if token else {}
    try:
        response = session.request(
            method, url, headers=headers, timeout=(10, 20), allow_redirects=False
        )
    except requests.RequestException as exc:
        # Exception messages or response bodies could include credentials.
        print(label + ": NETWORK_ERROR (" + type(exc).__name__ + ")", flush=True)
        return None
    print(label + ": HTTP " + str(response.status_code), flush=True)
    return response.status_code


def token_format_summary(token):
    return {
        "empty": not token,
        "starts_with_hf_": token.startswith("hf_"),
        "contains_whitespace": any(c.isspace() for c in token),
        "contains_non_ascii": any(ord(c) > 127 for c in token),
        "contains_control_char": any(ord(c) < 32 or ord(c) == 127 for c in token),
        "contains_quote_or_backtick": any(c in "\"'`" for c in token),
        "contains_other_symbol": any(not c.isascii() or not (c.isalnum() or c == "_") for c in token),
    }


def read_token_gui():
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        value = simpledialog.askstring(
            "Hugging Face token",
            "Paste the complete hf_ token here (Ctrl+V), then click OK.",
            show="*", parent=root,
        )
    finally:
        root.destroy()
    if value is None:
        raise EOFError
    return value.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true",
                        help="Enter token in a masked window instead of the terminal.")
    parser.add_argument("--save-login", action="store_true",
                        help="Save validated token using the official Hugging Face login cache.")
    args = parser.parse_args()
    print("Python:", sys.executable)
    print("Requests:", requests.__file__)
    names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "HF_TOKEN",
             "HF_ENDPOINT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")
    print("Environment variable presence:",
          ", ".join(name + "=" + str(bool(os.environ.get(name))) for name in names))
    session = requests.Session()
    # Prevent implicit .netrc credentials. Retain normal proxy/CA handling.
    session.auth = lambda request: request
    endpoint = "https://huggingface.co/api/whoami-v2"
    check(session, "NO_TOKEN", endpoint)
    check(session, "INVALID_TEST_TOKEN", endpoint, "diagnostic_invalid_token")
    token = read_token_gui() if args.gui else getpass.getpass("Paste Hugging Face token (hidden): ").strip()
    if not re.fullmatch(r"hf_[A-Za-z0-9]+", token):
        print("TOKEN_FORMAT_INVALID: copy the complete hf_ token, without quotes or spaces.")
        for name, value in token_format_summary(token).items():
            print("FORMAT_" + name.upper() + ": " + str(value))
        print("Input contents were NOT printed, saved, or sent.")
        session.close()
        return 2
    status = check(session, "YOUR_TOKEN", endpoint, token)
    if status == 200:
        print("AUTH_VALID: token accepted. Token has NOT been saved.")
        check(session, "PALIGEMMA_ACCESS",
              "https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer_config.json",
              token, method="HEAD")
        if args.save_login:
            from huggingface_hub import login

            try:
                login(token=token, add_to_git_credential=False)
            except Exception as exc:
                print("LOGIN_SAVE_FAILED: " + type(exc).__name__)
                session.close()
                return 1
            print("LOGIN_SAVED: official Hugging Face login cache updated.")
    elif status == 401:
        print("AUTH_REJECTED: check that the token is correct and still active.")
    elif status == 400:
        print("BAD_REQUEST: token validity is UNCONFIRMED; compare baseline results.")
    else:
        print("AUTH_UNCONFIRMED: share the status lines only.")
    session.close()
    return 0 if status == 200 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; token was not saved.")
        sys.exit(130)
