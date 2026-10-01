from datetime import date
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "check_contributor", Path(__file__).resolve().parents[1] / "scripts/check_contributor.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class FakeGitHub:
    def __init__(self, created="2020-01-01", contributions=(), comments=()):
        self.created = created
        self.contributions = contributions
        self.comments = comments
        self.calls = []

    def __call__(self, path, payload=None, method=None):
        self.calls.append((path, payload, method))
        if path.startswith("/users/"):
            return {"created_at": self.created + "T00:00:00Z", "type": "User"}
        if path == "/graphql":
            variables = payload["variables"]
            start, end = variables["from"][:10], variables["to"][:10]
            count = sum(n for day, n in self.contributions if start <= day <= end)
            return {"data": {"user": {
                "createdAt": self.created + "T00:00:00Z",
                "contributionsCollection": {"contributionCalendar": {"totalContributions": count}},
            }}}
        if payload is None:
            return self.comments
        return {}


class TestContributorEligibility(unittest.TestCase):
    today = date(2026, 10, 1)
    pr = {"number": 7, "user": {"login": "contributor"}}

    def test_exact_thresholds_and_recent_activity(self):
        for created, count, expected in [("2026-04-01", 100, True),
                                          ("2026-04-02", 100, False),
                                          ("2026-04-01", 99, False)]:
            with self.subTest(created=created, count=count):
                github = FakeGitHub(created, [("2026-09-01", count), ("2026-09-02", 1000)])
                _, cutoff, total, eligible = check.eligibility("contributor", self.today, github)
                self.assertEqual(cutoff, date(2026, 9, 1))
                self.assertEqual(total, count)
                self.assertEqual(eligible, expected)

    def test_history_windows_do_not_overlap(self):
        github = FakeGitHub(contributions=[("2025-09-02", 40), ("2025-09-01", 60)])
        self.assertTrue(check.eligibility("contributor", self.today, github)[-1])
        self.assertEqual(len(github.calls), 3)
        self.assertEqual(github.calls[2][1]["variables"]["to"], "2025-09-01T23:59:59Z")

    def test_month_ends_and_leap_years(self):
        self.assertEqual(check.months_before(date(2024, 3, 31), 1), date(2024, 2, 29))
        self.assertEqual(check.months_before(date(2025, 3, 31), 1), date(2025, 2, 28))
        github = FakeGitHub(contributions=[("2023-03-01", 100)])
        check.eligibility("contributor", date(2024, 3, 28), github)
        self.assertEqual(github.calls[1][1]["variables"]["from"], "2023-03-01T00:00:00Z")

    def test_pass_replies_and_does_not_close(self):
        github = FakeGitHub(contributions=[("2026-08-01", 100)])
        check.process_pr("owner/repo", self.pr, self.today, github)
        writes = [c for c in github.calls if c[2]]
        self.assertEqual(len(writes), 1)
        self.assertIn("🟢 **Contributor eligibility requirements met**", writes[0][1]["body"])
        self.assertIn(check.POLICY, writes[0][1]["body"])

    def test_fail_replies_then_closes(self):
        github = FakeGitHub("2026-09-01")
        check.process_pr("owner/repo", self.pr, self.today, github)
        writes = [c for c in github.calls if c[2]]
        self.assertEqual([c[2] for c in writes], ["POST", "PATCH"])
        self.assertIn("🔴 **Contributor eligibility requirements not met**", writes[0][1]["body"])
        self.assertIn("CONTRIBUTING.md#contributor-eligibility", writes[0][1]["body"])
        self.assertEqual(writes[1][1], {"state": "closed"})

    def test_rerun_updates_existing_bot_reply(self):
        comment = {"id": 42, "body": check.MARKER, "user": {"login": "github-actions[bot]"}}
        github = FakeGitHub(contributions=[("2026-08-01", 100)], comments=[comment])
        check.process_pr("owner/repo", self.pr, self.today, github)
        writes = [c for c in github.calls if c[2]]
        self.assertEqual(writes[0][0], "/repos/owner/repo/issues/comments/42")
        self.assertEqual(writes[0][2], "PATCH")

    def test_api_failure_does_not_close(self):
        calls = []
        def unavailable(*args):
            calls.append(args)
            raise RuntimeError("API unavailable")
        with self.assertRaises(RuntimeError):
            check.process_pr("owner/repo", self.pr, self.today, unavailable)
        self.assertEqual(len(calls), 1)

    def test_bot_is_not_treated_as_an_api_failure(self):
        def bot_profile(path, *args):
            self.assertEqual(path, "/users/dependabot%5Bbot%5D")
            return {"created_at": "2019-04-16T00:00:00Z", "type": "Bot"}
        created, _, count, eligible = check.eligibility("dependabot[bot]", self.today, bot_profile)
        self.assertEqual(created, date(2019, 4, 16))
        self.assertEqual(count, 0)
        self.assertFalse(eligible)

    def test_documentation_matches_quoted_policy(self):
        text = (Path(__file__).resolve().parents[2] / "CONTRIBUTING.md").read_text()
        self.assertIn("> " + check.POLICY, text)
