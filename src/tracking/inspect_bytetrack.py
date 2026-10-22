
import inspect
from importlib.metadata import PackageNotFoundError, version


def main():
    print("DAY 23 - BYTETRACK API INSPECTION")
    print("=" * 55)

    try:
        package_version = version("bytetracker")
        print(f"Package version: {package_version}")
    except PackageNotFoundError:
        raise RuntimeError(
            "bytetracker is not installed in this Python environment."
        )

    try:
        import bytetracker
        from bytetracker import BYTETracker
    except Exception as exc:
        raise RuntimeError(
            f"Could not import BYTETracker: {exc}"
        ) from exc

    print(f"Module path: {bytetracker.__file__}")
    print(f"BYTETracker class: {BYTETracker}")
    print()

    print("CONSTRUCTOR SIGNATURE")
    print(inspect.signature(BYTETracker))
    print()

    print("UPDATE SIGNATURE")
    print(inspect.signature(BYTETracker.update))
    print()

    print("UPDATE SOURCE")
    try:
        print(inspect.getsource(BYTETracker.update))
    except (OSError, TypeError) as exc:
        print(f"Could not retrieve source: {exc}")

    print()
    print("DAY 23 INSPECTION COMPLETE")
    print("Use the displayed API to determine the exact input format.")


if __name__ == "__main__":
    main()
