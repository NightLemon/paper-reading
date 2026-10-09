import tempfile
import unittest
from pathlib import Path

from check_reading_status import inspect


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / "docs"
        paper = self.docs / "topics" / "distributed-systems" / "2014-raft" / "README.md"
        paper.parent.mkdir(parents=True)
        paper.write_text('---\nstatus: done\nread_date: "2026-09-02"\n---\n' + "\n".join(f"## {i}. section\nSubstantive fixture text." for i in range(1, 7)), encoding="utf-8")
        (paper.parent.parent / "README.md").write_text("| [Raft](2014-raft/README.md) | 2014 | ATC | ✅ 笔记完成 | six sections |\n", encoding="utf-8")
        (self.docs / "index.md").write_text("| [Raft](topics/distributed-systems/2014-raft/README.md) | DS | 2014 | ATC | ✅ 笔记完成 | 5 |\n| [distributed-systems](topics/distributed-systems/README.md) | DS | 1 | 1 |\n", encoding="utf-8")
        (self.docs / "reading-list.md").write_text("| Raft | DS | [note](topics/distributed-systems/2014-raft/README.md) | 2026-09-02 |\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def change(self, relative, old, new):
        path = self.docs / relative
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    def test_consistent_notes_pass(self):
        self.assertEqual(inspect(self.root)["status"], "pass")

    def test_stale_raft_status_and_count_fail(self):
        self.change("index.md", "✅ 笔记完成", "🚧 已派工单")
        self.change("index.md", "| DS | 1 | 1 |", "| DS | 0 | 1 |")
        errors = inspect(self.root)["errors"]
        self.assertTrue(any("status does not match" in error for error in errors))
        self.assertTrue(any("counts should be 1/1" in error for error in errors))

    def test_glued_table_rows_fail(self):
        path = self.docs / "index.md"
        text = path.read_text(encoding="utf-8")
        topic = text.splitlines()[-1]
        path.write_text(text + topic + topic + "\n", encoding="utf-8")
        self.assertTrue(any("malformed topic row" in error for error in inspect(self.root)["errors"]))

    def test_promoted_skeleton_and_missing_sections_fail(self):
        self.change("topics/distributed-systems/2014-raft/README.md", "## 6. section", "<待精读>")
        errors = inspect(self.root)["errors"]
        self.assertTrue(any("six sections" in error for error in errors))
        self.assertTrue(any("skeleton" in error for error in errors))

    def test_stale_topic_and_reading_queue_fail(self):
        self.change("topics/distributed-systems/README.md", "✅ 笔记完成", "🚧 已派工单")
        self.change("reading-list.md", "[note]", "[骨架页]")
        errors = inspect(self.root)["errors"]
        self.assertTrue(any("status differs" in error for error in errors))
        self.assertTrue(any("still marked as pending" in error for error in errors))

    def test_duplicate_and_missing_paper_rows_fail(self):
        path = self.docs / "index.md"
        text = path.read_text(encoding="utf-8")
        path.write_text(text + text.splitlines()[0] + "\n", encoding="utf-8")
        self.assertTrue(any("duplicate paper" in error for error in inspect(self.root)["errors"]))
        path.write_text(text.splitlines()[1] + "\n", encoding="utf-8")
        self.assertTrue(any("missing paper" in error for error in inspect(self.root)["errors"]))

    def test_ambiguous_visible_status_fails(self):
        self.change("index.md", "✅ 笔记完成", "✅ 笔记完成（待精读）")
        self.assertTrue(any("status does not match" in error for error in inspect(self.root)["errors"]))

    def test_malformed_frontmatter_status_has_a_diagnostic(self):
        self.change("topics/distributed-systems/2014-raft/README.md", "status: done", "status: []")
        self.assertTrue(any("invalid note status" in error for error in inspect(self.root)["errors"]))

    def test_toread_does_not_become_complete_from_six_template_headings(self):
        self.change("topics/distributed-systems/2014-raft/README.md", "status: done", "status: to-read")
        self.change("index.md", "✅ 笔记完成", "📥 待读（已派工单）")
        self.change("index.md", "| DS | 1 | 1 |", "| DS | 0 | 1 |")
        self.change("topics/distributed-systems/README.md", "✅ 笔记完成", "🚧 已派工单")
        result = inspect(self.root)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["completed_count"], 0)


if __name__ == "__main__":
    unittest.main()
