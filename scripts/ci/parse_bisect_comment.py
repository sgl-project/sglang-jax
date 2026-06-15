"""Parse /auto-bisect slash command from a GitHub issue comment body.

Reads COMMENT_BODY from env, writes to GITHUB_OUTPUT:
  valid=true|false
  run_id=<digits or empty>

Strict syntax: /auto-bisect [run_id]
  /auto-bisect          -> valid=true, run_id=
  /auto-bisect 12345    -> valid=true, run_id=12345
  anything else         -> valid=false, run_id=
"""

import os
import re
import sys

PATTERN = re.compile(r"^/auto-bisect(?:\s+(\d+))?\s*$")


def main() -> int:
    body = os.environ.get("COMMENT_BODY", "").strip()
    output_path = os.environ["GITHUB_OUTPUT"]

    match = PATTERN.match(body)
    valid = "true" if match else "false"
    run_id = (match.group(1) or "") if match else ""

    with open(output_path, "a", encoding="utf-8") as fh:
        fh.write(f"valid={valid}\n")
        fh.write(f"run_id={run_id}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
