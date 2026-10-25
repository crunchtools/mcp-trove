"""Office documents, spreadsheets and presentations, and what gets indexed (0.15.0)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mcp_trove_crunchtools import config as config_mod
from mcp_trove_crunchtools import database as db
from mcp_trove_crunchtools import documents
from mcp_trove_crunchtools import extractor as extractor_mod
from mcp_trove_crunchtools.config import get_config
from mcp_trove_crunchtools.errors import ExtractionError
from mcp_trove_crunchtools.extractor import extract_text
from mcp_trove_crunchtools.indexer import index_path_async

if TYPE_CHECKING:
    import sqlite3

FIXTURES = Path(__file__).parent / "fixtures" / "documents"
LETTER = "Installed in the 2008 Jeep Grand Cherokee on 3 March 2024."
BUDGET_ROW = "Duralast battery\t189.99\t2024-03-03"


class TestReaders:
    @pytest.mark.parametrize("name", ["letter.doc", "letter.odt", "letter.rtf"])
    def test_documents(self, name: str) -> None:
        text = extract_text(FIXTURES / name)
        assert "Duralast Gold battery receipt" in text
        assert LETTER in text

    @pytest.mark.parametrize("name", ["budget.ods", "budget.xls", "budget.xlsx"])
    def test_spreadsheets_keep_rows_and_dates(self, name: str) -> None:
        text = extract_text(FIXTURES / name)
        assert text.startswith("[Sheet] budget\nItem\tCost\tDate")
        assert BUDGET_ROW in text

    @pytest.mark.parametrize("name", ["deck.odp", "deck.pptx"])
    def test_presentations(self, name: str) -> None:
        text = extract_text(FIXTURES / name)
        assert text.startswith("[Slide 1]\nRHEL roadmap\nImage mode for every workload")
        assert "Mention bootc" in text
        assert "<number>" not in text

    def test_sheet_keeps_its_leading_rows(self, tmp_path: Path) -> None:
        import openpyxl

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        for row in range(1, 301):
            sheet.append([f"row {row}", row])
        workbook.save(tmp_path / "long.xlsx")

        lines = extract_text(tmp_path / "long.xlsx").splitlines()

        assert len(lines) == 1 + documents.SHEET_MAX_ROWS
        assert lines[-1] == f"row {documents.SHEET_MAX_ROWS}\t{documents.SHEET_MAX_ROWS}"

    def test_csv_keeps_its_leading_lines(self, tmp_path: Path) -> None:
        (tmp_path / "big.csv").write_text("".join(f"{n},x\n" for n in range(500)))

        assert len(extract_text(tmp_path / "big.csv").splitlines()) == documents.SHEET_MAX_ROWS

    def test_encrypted_odf_says_so(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked.odt"
        with zipfile.ZipFile(locked, "w") as archive:
            archive.writestr("META-INF/manifest.xml", "<manifest:encryption-data/>")
            archive.writestr("content.xml", b"\x00garbled")

        with pytest.raises(ExtractionError, match="password-protected"):
            extract_text(locked)

    @pytest.mark.parametrize("name", ["bomb.docx", "bomb.xlsx", "bomb.pptx", "bomb.odt"])
    def test_package_that_unpacks_too_far_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setattr(documents, "PACKAGE_MAX_UNZIPPED", 1000)
        with zipfile.ZipFile(tmp_path / name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("content.xml", b"0" * 5000)

        with pytest.raises(ExtractionError, match="over the 1000 byte limit"):
            extract_text(tmp_path / name)

    def test_broken_document_fails_just_that_file(self, tmp_path: Path) -> None:
        (tmp_path / "broken.xlsx").write_bytes(b"not a zip")

        with pytest.raises(ExtractionError, match=r"broken\.xlsx"):
            extract_text(tmp_path / "broken.xlsx")

    def test_word_fields_keep_their_result_only(self) -> None:
        raw = 'See \x13 HYPERLINK "https://x" \x14the site\x15 now\rCell\x07Next'

        assert documents._clean_word_text(raw) == "See the site now\nCell\tNext"


class TestCategories:
    def test_unknown_category_fails_at_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_INDEX_TYPES", "pdf,spreadsheet")
        config_mod._config = None

        with pytest.raises(ValueError, match="unknown categories spreadsheet"):
            get_config()

    def test_bad_catch_all_fails_at_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TROVE_INDEX_OTHER", "everything")
        config_mod._config = None

        with pytest.raises(ValueError, match="skip or name"):
            get_config()


class TestNameEntries:
    @pytest.fixture
    def name_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_config(), "index_other", "name")

    async def test_other_files_are_found_by_name_without_being_read(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, name_mode: None
    ) -> None:
        drawing = tmp_path / "Garage plan.dwg"
        drawing.write_bytes(b"secret bytes")
        drawing.chmod(0)  # reading it would fail the file

        try:
            results = await index_path_async(tmp_path)
        finally:
            drawing.chmod(0o600)

        assert [r["status"] for r in results] == ["indexed"]
        row = db.query_one("SELECT file_type, checksum FROM files")
        assert row is not None
        assert row["file_type"] == "name"
        assert row["checksum"].startswith("name:")
        content = db.query_one("SELECT content FROM chunks")
        assert content is not None
        assert "[File] Garage plan.dwg" in content["content"]
        assert "[Type] dwg" in content["content"]

    async def test_file_too_big_to_read_gets_a_name_entry(
        self,
        tmp_path: Path,
        in_memory_db: sqlite3.Connection,
        name_mode: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(extractor_mod, "MAX_FILE_SIZE", 10)
        (tmp_path / "manual.txt").write_text("far more than ten bytes of text")

        await index_path_async(tmp_path)

        assert db.query_one("SELECT file_type FROM files") == {"file_type": "name"}

    async def test_switching_a_category_on_reads_the_file(
        self,
        tmp_path: Path,
        in_memory_db: sqlite3.Connection,
        name_mode: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "page.html").write_text("<p>Valvoline oil change</p>")
        await index_path_async(tmp_path)
        assert db.query_one("SELECT file_type FROM files") == {"file_type": "name"}

        config = get_config()
        monkeypatch.setattr(config, "index_types", config.index_types | {"web"})
        results = await index_path_async(tmp_path)

        assert [r["status"] for r in results] == ["indexed"]
        assert db.query_one("SELECT file_type FROM files") == {"file_type": "text"}

    async def test_own_database_is_not_indexed(
        self, tmp_path: Path, name_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db.get_db(str(tmp_path / "trove.db"))
        (tmp_path / "notes.txt").write_text("keep")

        results = await index_path_async(tmp_path)

        assert [Path(str(r["path"])).name for r in results] == ["notes.txt"]


class TestPurge:
    async def test_switching_a_category_off_removes_its_entries(
        self, tmp_path: Path, in_memory_db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = get_config()
        monkeypatch.setattr(config, "index_types", config.index_types | {"data"})
        (tmp_path / "export.json").write_text('{"car": "Jeep"}')
        await index_path_async(tmp_path)
        monkeypatch.setattr(config, "index_types", config.index_types - {"data"})

        results = await index_path_async(tmp_path)

        assert [r["reason"] for r in results] == ["excluded"]
        assert db.query("SELECT id FROM files") == []
