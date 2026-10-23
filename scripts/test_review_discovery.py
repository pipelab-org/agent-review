import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
from scripts import review_discovery as discovery


class DiscoveryTests(unittest.TestCase):
    def test_paths_are_literal_repository_paths(self):
        self.assertEqual(["src/a.py"], discovery.requested_paths({"source_requests": ["src/a.py", "src/a.py"]}))
        for value in (None, "src/a.py", ["../a"], ["/a"], ["a//b"], ["a/./b"],
                      ["https://api.vendor.example/a"], ["a\\b"], ["a\n"], [False]):
            with self.subTest(value=value), self.assertRaises(discovery.DiscoveryAbort):
                discovery.requested_paths({"source_requests": value})

    def test_source_read_is_revision_bound_and_rejects_nonfiles(self):
        repo = Mock()
        repo.get_contents.return_value = NS(type="file", encoding="base64", raw_data={}, decoded_content=b"guard()")
        self.assertEqual("guard()", discovery.read_source(repo, "src/a.py", "a" * 40))
        repo.get_contents.assert_called_once_with("src/a.py", ref="a" * 40)
        for item in ([], NS(type="dir"), NS(type="symlink"), NS(type="file", encoding="none"),
                     NS(type="file", encoding="base64", raw_data={"target": "elsewhere"}),
                     NS(type="file", encoding="base64", raw_data={"submodule_git_url": "remote"}),
                     NS(type="file", encoding="base64", raw_data={}, decoded_content=b"\xff")):
            repo.get_contents.return_value = item
            with self.assertRaises(discovery.Unavailable):
                discovery.read_source(repo, "src/a.py", "a" * 40)
        for failure in (OSError("network"), type("Failure", (Exception,), {"status": 500})(), Exception("unknown")):
            repo.get_contents.side_effect = failure
            with self.assertRaises(discovery.DiscoveryAbort) as raised:
                discovery.read_source(repo, "src/a.py", "a" * 40)
            self.assertNotIsInstance(raised.exception, discovery.Unavailable)
        repo.get_contents.side_effect = type("NotFound", (Exception,), {"status": 404})()
        with self.assertRaises(discovery.Unavailable):
            discovery.read_source(repo, "src/a.py", "a" * 40)

    def test_a_colon_is_a_legal_filename_character_but_urls_stay_invalid(self):
        self.assertEqual(["docs/a:b.md"], discovery.requested_paths({"source_requests": ["docs/a:b.md"]}))
        for value in ("https://api.vendor.example/a", "file:///etc/passwd", "//host/share"):
            with self.subTest(value=value), self.assertRaises(discovery.DiscoveryAbort):
                discovery.requested_paths({"source_requests": [value]})

    def test_unloaded_head_content_is_read_but_loaded_and_deleted_files_are_not(self):
        read = Mock(return_value="from head")
        deleted = NS(name="DELETED")
        cases = (("loaded", NS(head_file="loaded", edit_type=NS(name="MODIFIED")), "loaded", 0),
                 ("unloaded", NS(head_file="", filename="a.py", edit_type=NS(name="MODIFIED")), "from head", 1),
                 ("unknown type", NS(head_file="", filename="a.py"), "from head", 1),
                 ("deleted", NS(head_file="", filename="a.py", edit_type=deleted), "", 0))
        for name, file, expected, reads in cases:
            read.reset_mock()
            with self.subTest(name):
                self.assertEqual(expected, discovery.head_content(file, read))
                self.assertEqual(reads, read.call_count)

    def test_later_empty_output_cannot_silently_discard_initial_candidates(self):
        first = {"review": {"key_issues_to_review": [{"issue_header": "original"}], "security_concerns": "Original concern"}}
        second = {"review": {"key_issues_to_review": [], "security_concerns": "No"}}
        result = discovery.merge(first, second)
        self.assertEqual(first, result)
        self.assertEqual([], second["review"]["key_issues_to_review"])

    def test_new_candidates_and_distinct_concerns_reach_challenge(self):
        first = {"review": {"key_issues_to_review": [{"issue_header": "first"}], "security_concerns": "First"}}
        second = {"review": {"key_issues_to_review": [{"issue_header": "first"}, {"issue_header": "second"}], "security_concerns": "Second"}}
        result = discovery.merge(first, second)
        self.assertEqual(2, len(result["review"]["key_issues_to_review"]))
        self.assertEqual("First\n\nSecond", result["review"]["security_concerns"])
