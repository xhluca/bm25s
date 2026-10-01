"""Check PR-author eligibility using trusted code and GitHub profile metadata."""
import calendar
from datetime import date, timedelta
import json
import os
from pathlib import Path
import urllib.request
from urllib.parse import quote

POLICY = (
    "Pull request authors must have a GitHub account at least six calendar months old "
    "and at least 100 GitHub contributions dated on or before the date one calendar "
    "month before the check."
)
MARKER = "<!-- bm25s-contributor-eligibility -->"
QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      contributionCalendar { totalContributions }
    }
  }
}
"""


def months_before(day, months):
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def api(path, payload=None, method=None):
    request = urllib.request.Request(
        os.environ.get("GITHUB_API_URL", "https://api.github.com") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if isinstance(result, dict) and result.get("errors"):
        raise RuntimeError("GitHub could not verify contribution history")
    return result


def eligibility(login, today, request=api):
    cutoff = months_before(today, 1)
    end = cutoff
    total = 0
    profile = request("/users/" + quote(login, safe=""))
    created = date.fromisoformat(profile["created_at"][:10])
    # GitHub App bots do not have a user contribution calendar.
    if profile["type"] == "Bot":
        return created, cutoff, 0, False
    # GitHub limits each contribution-history query to at most one year.
    while end >= created:
        start = months_before(end, 12) + timedelta(days=1)
        result = request("/graphql", {"query": QUERY, "variables": {
            "login": login, "from": start.isoformat() + "T00:00:00Z",
            "to": end.isoformat() + "T23:59:59Z",
        }})
        user = result["data"]["user"]
        if user is None:
            raise RuntimeError("GitHub profile is unavailable; eligibility was not verified")
        total += user["contributionsCollection"]["contributionCalendar"]["totalContributions"]
        if total >= 100:
            break
        end = start - timedelta(days=1)
    return created, cutoff, total, created <= months_before(today, 6) and total >= 100


def process_pr(repository, pr, today, request=api):
    login = pr["user"]["login"]
    created, cutoff, total, eligible = eligibility(login, today, request)
    policy_url = f"https://github.com/{repository}/blob/HEAD/CONTRIBUTING.md#contributor-eligibility"
    result = "met" if eligible else "not met"
    body = (
        f"{MARKER}\nContributor eligibility requirements **{result}** for @{login}.\n\n"
        f"- Account created: {created}; age: {(today - created).days} days "
        f"(required: at least six calendar months).\n"
        f"- Contributions dated on or before {cutoff}: {total}"
        f"{' or more' if total >= 100 else ''} (required: at least 100).\n\n"
        f"> {POLICY}\n\nSee [the contributor eligibility policy]({policy_url})."
    )
    if not eligible:
        body += (
            "\n\nClosing this PR because these requirements were not met. "
            "This policy is intended to reduce spam from AI agents; it is not an "
            "assessment of the quality of your change."
        )
    number = pr["number"]
    # Find the existing bot reply across all pages before posting another one.
    page = 1
    while True:
        comments = request(f"/repos/{repository}/issues/{number}/comments?per_page=100&page={page}")
        comment = next((c for c in comments if c["user"]["login"] == "github-actions[bot]"
                        and MARKER in c["body"]), None)
        if comment:
            request(f"/repos/{repository}/issues/comments/{comment['id']}",
                    {"body": body}, "PATCH")
            break
        if len(comments) < 100:
            request(f"/repos/{repository}/issues/{number}/comments", {"body": body}, "POST")
            break
        page += 1
    if not eligible:
        request(f"/repos/{repository}/pulls/{number}", {"state": "closed"}, "PATCH")
    print(f"PR #{number}: @{login}, requirements {result}")


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    today = date.fromisoformat(os.environ["CHECK_DATE"])
    if "pull_request" in event:
        prs = [event["pull_request"]]
    else:
        # Manual dispatch can apply the policy to all currently open PRs.
        prs = []
        page = 1
        while True:
            batch = api(f"/repos/{repository}/pulls?state=open&per_page=100&page={page}")
            prs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
    for pr in prs:
        if pr["state"] == "open":
            process_pr(repository, pr, today)


if __name__ == "__main__":
    main()
