# Agent skills

`adabmdca/` is an [Agent Skill](https://agentskills.io) that teaches coding agents with a shell (Claude Code, Codex and others) to run adabmDCA: preparing alignments, training, sampling, scoring, mutation scans, contacts, entropy, and checking that a model is healthy.

`adabmdca-steering/` covers steered generation from Python: substitution windows around reference sequences, allowed letters, Hamming biases, scoring callbacks, low-temperature sampling and the steered entropy, and when to use plain MCMC or PTT for them. Install it next to the first one.

## Install

Copy or link the folder where your agent looks for skills:

```bash
# Claude Code, for all projects
mkdir -p ~/.claude/skills && cp -r skills/adabmdca skills/adabmdca-steering ~/.claude/skills/

# Claude Code, for one project
mkdir -p .claude/skills && cp -r /path/to/adabmDCApy/skills/adabmdca* .claude/skills/

# Codex
mkdir -p ~/.codex/skills && cp -r skills/adabmdca skills/adabmdca-steering ~/.codex/skills/
```

The skill expects `adabmDCA` to be installed in the environment the agent uses (`python -m pip install 'adabmDCA[cpu]'`).

For clients without a shell, such as Claude Desktop, use the [agentDCA](https://github.com/rossetl/agentDCA) MCP server instead.

## Keep it in sync

The skill quotes command options, defaults and output columns. When a change to the package renames or adds one of those, update `adabmdca/SKILL.md`, the matching file in `adabmdca/references/`, or `adabmdca-steering/SKILL.md` in the same pull request.
