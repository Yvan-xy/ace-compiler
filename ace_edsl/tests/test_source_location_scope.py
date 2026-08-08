"""Source-location cache ownership regressions."""

from ace_edsl.base_dsl import loc


class _FakeGlobScope:
    def __init__(self, file_id: int) -> None:
        self.file_id = file_id
        self.registered: list[str] = []

    def register_file(self, filename: str) -> int:
        self.registered.append(filename)
        return self.file_id


def test_file_id_cache_is_scoped_to_air_module() -> None:
    filename = "retained_ckks_conformance.py"
    original_scope = loc.get_glob_scope()
    first = _FakeGlobScope(3)
    second = _FakeGlobScope(11)
    loc.clear_file_cache()

    try:
        loc.set_glob_scope(first)
        assert loc.register_file(filename) == 3
        assert loc.register_file(filename) == 3
        assert first.registered == [filename]

        loc.set_glob_scope(second)
        assert loc.register_file(filename) == 11
        assert second.registered == [filename]
    finally:
        loc.set_glob_scope(original_scope)
        loc.clear_file_cache()
