"""支持 python -m wlxq_bot 调用 CLI。"""

from wlxq_bot.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
