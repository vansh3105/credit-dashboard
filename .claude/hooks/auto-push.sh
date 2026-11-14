#!/bin/bash
# Auto-commit and push any changes to origin/main at the end of each Claude turn.
cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0
[ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = "main" ] || exit 0

if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -q -m "Auto-update $(date '+%Y-%m-%d %H:%M')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || exit 0
fi

git pull -q --rebase origin main 2>/dev/null || { git rebase --abort 2>/dev/null; echo "{\"systemMessage\": \"Auto-push skipped: GitHub has changes that conflict with local ones. Ask Claude to resolve.\"}"; exit 0; }

if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
  if git push -q origin main 2>/tmp/claude-autopush.err; then
    echo '{"systemMessage": "Changes pushed to GitHub (origin/main)."}'
  else
    echo "{\"systemMessage\": \"Auto-push to GitHub failed: $(head -c 200 /tmp/claude-autopush.err | tr '\n\"' '  ')\"}"
  fi
fi
