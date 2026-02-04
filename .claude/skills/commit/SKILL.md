---
name: commit
description: Create a git commit with a well-structured commit message based on staged changes
disable-model-invocation: true
allowed-tools: Bash,Read,Glob,Grep,AskUserQuestion
---

# Commit Skill

Create a git commit with a descriptive, well-structured commit message.

## Steps

1. **Pull latest changes:**
   - Run `git pull` to fetch and merge latest changes from remote
   - If there are conflicts, inform the user and stop

2. **Check git status and changes:**
   - Run `git status` to see staged and unstaged changes
   - Run `git diff --cached` to see staged changes (this is what will be committed)
   - Only analyze staged changes for the commit message; ignore unstaged changes

3. **Review recent commits for style:**
   - Run `git log --oneline -10` to see recent commit message style
   - Match the repository's commit message conventions

4. **Analyze the changes:**
   - Understand what files were modified and why
   - Read relevant files if needed to understand the context
   - Categorize the change type: fix, feat, docs, refactor, test, chore, etc.

5. **Propose a commit message:**
   - Write a clear, concise subject line (50 chars or less if possible)
   - Add a body explaining the "why" not just the "what"
   - Group related changes logically
   - Use the format:
     ```
     <type>: <subject>

     <body with details>
     ```

6. **Get user approval:**
   - Present the proposed commit message to the user
   - Use AskUserQuestion with these options:
     - "Yes" - proceed with the commit
     - "Yes, and don't ask again" - proceed and remember for this repo
     - "No" - cancel the commit

7. **Create the commit:**
   - Stage any requested files
   - Create the commit with the approved message
   - Use HEREDOC format for multi-line messages:
     ```bash
     git commit -m "$(cat <<'EOF'
     Commit message here.

     Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
     EOF
     )"
     ```

8. **Ask about pushing:**
   - After successful commit, use AskUserQuestion with these options:
     - "Yes" - push to remote
     - "No" - don't push

## Rules

- NEVER commit files that may contain secrets (.env, credentials.json, API keys)
- NEVER use `git commit --amend` unless explicitly requested by the user
- NEVER skip pre-commit hooks unless explicitly requested
- Always include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>` at the end
- If pre-commit hooks fail, fix the issues and create a NEW commit (don't amend)
