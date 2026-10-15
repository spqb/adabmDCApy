# Agent skills

`adabmdca/` is an [Agent Skill](https://agentskills.io) that teaches coding agents with a shell (Claude Code, Codex and others) to run adabmDCA: preparing alignments, training, sampling, scoring, mutation scans, contacts, entropy, and checking that a model is healthy.

## Install

Copy or link the folder where your agent looks for skills:

```bash
# Claude Code, for all projects
mkdir -p ~/.claude/skills && cp -r skills/adabmdca ~/.claude/skills/

# Claude Code, for one project
mkdir -p .claude/skills && cp -r /path/to/adabmDCApy/skills/adabmdca .claude/skills/

# Codex
mkdir -p ~/.codex/skills && cp -r skills/adabmdca ~/.codex/skills/
```

The skill expects `adabmDCA` to be installed in the environment the agent uses (`python -m pip install 'adabmDCA[cpu]'`).

For clients without a shell, such as Claude Desktop, use the [agentDCA](https://github.com/rossetl/agentDCA) MCP server instead.

## Keep it in sync

The skill quotes command options, defaults and output columns. When a change to the package renames or adds one of those, update `adabmdca/SKILL.md` or the matching file in `adabmdca/references/` in the same pull request.
