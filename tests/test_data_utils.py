from pathlib import Path

import pytest

from usas_csv_auto_labeling import data_utils


@pytest.mark.parametrize("usas_tag_description_file_str",
                        [None, "data/data_utils/test_usas_mapper.yaml"])
@pytest.mark.parametrize("tags_to_filter_out", [None, set(["Z99"])])
def test_load_usas_mapper(usas_tag_description_file_str: str | None,
                          tags_to_filter_out: set[str] | None) -> None:
    usas_tag_description_file: Path | None = None
    if usas_tag_description_file_str is not None:
        usas_tag_description_file = Path(__file__).parent / usas_tag_description_file_str
    usas_mapper = data_utils.load_usas_mapper(usas_tag_description_file,
                                                              tags_to_filter_out)
    assert isinstance(usas_mapper, dict)
    assert len(usas_mapper) > 0

    if tags_to_filter_out is None and usas_tag_description_file is None:
        assert len(usas_mapper) == 222
        assert "Z99" in usas_mapper
    elif tags_to_filter_out is not None and usas_tag_description_file is None:
        assert len(usas_mapper) == 221
        assert "Z99" not in usas_mapper
    elif usas_tag_description_file is not None:
        assert len(usas_mapper) == 1
    
    assert "A1.1.1" in usas_mapper
    if usas_tag_description_file is None:
        expected_title_description = (
            "title: General actions, making etc. description: "
            "General/abstract terms relating to an activity/action "
            "(e.g. act, adventure, approach, arise); a characteristic/feature "
            "(e.g. absorb, attacking, automatically); "
            "aconstruction/craft and/or the action of constructing/crafting "
            "(e.g. arrange, assemble, bolts, boring, break)"
        )
        assert expected_title_description == usas_mapper["A1.1.1"]
    else:
        assert "title: General Test description: Test Case" == usas_mapper["A1.1.1"]

    assert "A.1" not in usas_mapper

def test_load_usas_mapper_with_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        data_utils.load_usas_mapper(Path(__file__).parent / "nonexistent.yaml", None)

def test_load_usas_mapper_with_directory_path() -> None:
    with pytest.raises(ValueError):
        data_utils.load_usas_mapper(Path(__file__).parent, None)

def test_load_usas_mapper_with_no_title() -> None:
    with pytest.raises(KeyError):
        no_title_usas_tag_description_file = Path(__file__).parent / "data/data_utils/test_usas_mapper_no_title.yaml"
        data_utils.load_usas_mapper(no_title_usas_tag_description_file, None)

def test_load_usas_mapper_with_no_description() -> None:
    with pytest.raises(KeyError):
        no_description_usas_tag_description_file = Path(__file__).parent / "data/data_utils/test_usas_mapper_no_description.yaml"
        data_utils.load_usas_mapper(no_description_usas_tag_description_file, None)

def test_load_usas_mapper_duplicate_key() -> None:
    with pytest.raises(KeyError):
        duplicate_key_usas_tag_description_file = Path(__file__).parent / "data/data_utils/test_usas_mapper_duplicate_key.yaml"
        data_utils.load_usas_mapper(duplicate_key_usas_tag_description_file, None)
        

