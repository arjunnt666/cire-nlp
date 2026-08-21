import io
import json
import unittest
from unittest import mock

import cire_engine as cire


class CireTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(cire.normalize("  What's UP?? "), "what is up")

    def test_meta_help(self):
        self.assertEqual(cire.detect_meta_intent("help"), "help")

    def test_topic_a(self):
        self.assertEqual(cire.detect_topic("topic a"), "domain_a")

    def test_resolve_rules(self):
        result = cire.resolve("rules for subject b")
        self.assertEqual(result["topic"], "domain_b")
        self.assertEqual(result["intent"], "rules")
        self.assertFalse(result["end_session"])

    def test_schema_ok(self):
        self.assertEqual(cire.CIREDiagnostics.validate_schema(), [])

    def test_topic_is_not_ranking(self):
        self.assertEqual(cire.detect_intent("topic a"), "info")
        self.assertEqual(cire.detect_intent("rules for topic b"), "rules")
        self.assertEqual(cire.detect_intent("top picks for topic a"), "ranking")
        result = cire.resolve("topic a")
        self.assertEqual(result["topic"], "domain_a")
        self.assertEqual(result["intent"], "info")

    def test_cli_argv(self):
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            code = cire.main(["rules for topic b"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["topic"], "domain_b")
        self.assertEqual(data["intent"], "rules")


if __name__ == "__main__":
    unittest.main()
