"""Minimal entry point with warning suppression."""

import os
import sys


def main():
    # if we haven't re-exec'd yet, do it with -W flag
    if not os.environ.get("_ULTRASYNC_WARNED"):
        os.environ["_ULTRASYNC_WARNED"] = "1"
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-W",
                "ignore::RuntimeWarning",
                "-m",
                "ultrasync._entry",
            ]
            + sys.argv[1:],
        )

    # now safe to import the real CLI
    from ultrasync.cli import main

    return main()


if __name__ == "__main__":
    sys.exit(main())
