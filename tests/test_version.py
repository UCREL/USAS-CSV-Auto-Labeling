import re

from usas_csv_auto_labeling import __version__


def test_version() -> None:
    version = __version__
    assert isinstance(version, str)
    assert re.match(r"^\d+\.\d+\.\d+$", version)