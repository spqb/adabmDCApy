# Agent skills

`adabmdca/` is an [Agent Skill](https://agentskills.io) that teaches coding agents with a shell (Claude Code, Codex and others) to run adabmDCA: preparing alignments, training, sampling, scoring, mutation scans, contacts, entropy, and checking that a model is healthy.

`adabmdca-steering/` covers steered generation from Python: substitution windows around reference sequences, allowed letters, Hamming biases, scoring callbacks, low-temperature sampling and the steered entropy, and when to use plain MCMC or PTT for them. `adabmdca-local-free-energy/` covers scoring sequences with the local free energy F = E − λ·CDE and fitting λ on equilibrium samples of the model. Install both next to the first one.

## Install

Copy or link the folder where your agent looks for skills:

```bash
# Claude Code, for all projects
mkdir -p ~/.claude/skills && cp -r skills/adabmdca* ~/.claude/skills/

# Claude Code, for one project
mkdir -p .claude/skills && cp -r /path/to/adabmDCApy/skills/adabmdca* .claude/skills/

# Codex, for all projects (run from the repository root)
mkdir -p ~/.agents/skills
for skill in adabmdca adabmdca-steering adabmdca-local-free-energy; do
  ln -s "$PWD/skills/$skill" "$HOME/.agents/skills/$skill"
done
```

For Codex in this repository, `.agents/skills/` already contains relative symlinks to the three skill folders, so no separate installation is needed. Codex discovers these skills automatically. The user-level links above make them available in other projects too; choose the scope you need to avoid duplicate entries. Symlinks keep skill changes in sync. If a skill does not appear after a change, restart Codex.

The skill expects `adabmDCA` to be installed in the environment the agent uses (`python -m pip install 'adabmDCA[cpu]'`).

For clients without a shell, such as Claude Desktop, use the [agentDCA](https://github.com/rossetl/agentDCA) MCP server instead.

## Keep it in sync

The skill quotes command options, defaults and output columns. When a change to the package renames or adds one of those, update `adabmdca/SKILL.md`, the matching file in `adabmdca/references/`, `adabmdca-steering/SKILL.md` or `adabmdca-local-free-energy/SKILL.md` in the same pull request.
