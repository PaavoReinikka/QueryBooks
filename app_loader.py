import sys


def _print_table_sources() -> None:
    from utils.loader_utils import load_environment
    from utils.retrieval_functions import (
        connect_from_env,
        format_table_sources,
        inspect_table_sources,
    )

    load_environment(".env")
    with connect_from_env() as conn:
        mapping = inspect_table_sources(conn)
    print(format_table_sources(mapping, markdown=False))


if __name__ == "__main__":
    if "--inspect-sources" in sys.argv:
        _print_table_sources()
    else:
        from apps.loader.app import launch_loader

        launch_loader()
