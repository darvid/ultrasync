"""Minimal entry point with warning suppression."""

import os
import sys


def main():
    # if we haven't re-exec'd yet, do it with -W flag
    if not os.environ.get("_GALAXYBRAIN_WARNED"):
        os.environ["_GALAXYBRAIN_WARNED"] = "1"
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-W",
                "ignore::RuntimeWarning",
                "-m",
                "galaxybrain._entry",
            ]
            + sys.argv[1:],
        )

    # now safe to import the real CLI
    from galaxybrain.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
