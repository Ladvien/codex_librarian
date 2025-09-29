# Claude Code Project Configuration

This directory contains Claude Code-specific configuration and context for the PDF to Markdown MCP project.

## Directory Structure

```
.claude/
├── README.md              # This file
├── PROJECT_CONTEXT.md     # Comprehensive project context (version controlled)
├── QUICK_REFERENCE.md     # Quick command reference (version controlled)
├── SESSION_NOTES.md       # Session history and notes (gitignored)
├── settings.local.json    # Local permissions and settings (gitignored)
├── agents/               # Custom agents for this project
└── output-styles/        # Custom output formatting (gitignored)
```

## Files Explanation

### Version Controlled (Shared with team)
- **PROJECT_CONTEXT.md**: Complete project overview, endpoints, common issues
- **QUICK_REFERENCE.md**: Most used commands and quick fixes
- **agents/**: Project-specific Claude Code agents

### Local Only (Gitignored)
- **SESSION_NOTES.md**: Your personal session notes and history
- **settings.local.json**: Your local permissions and preferences
- **output-styles/**: Your custom output formatting preferences

## Usage

When starting a new Claude Code session:
1. Claude Code will automatically read PROJECT_CONTEXT.md
2. Refer to QUICK_REFERENCE.md for common commands
3. Update SESSION_NOTES.md with important discoveries
4. Custom agents in agents/ will be available automatically

## Updating Context

To update project context for all team members:
```bash
# Edit the version controlled files
vim .claude/PROJECT_CONTEXT.md
vim .claude/QUICK_REFERENCE.md

# Commit changes
git add .claude/PROJECT_CONTEXT.md .claude/QUICK_REFERENCE.md
git commit -m "Update Claude Code project context"
```

To keep personal notes:
```bash
# Edit your local session notes (won't be committed)
vim .claude/SESSION_NOTES.md
```

## Best Practices

1. **Keep PROJECT_CONTEXT.md updated** with critical system info
2. **Document solutions** in SESSION_NOTES.md when solving issues
3. **Add frequent commands** to QUICK_REFERENCE.md
4. **Create custom agents** for repetitive tasks in agents/
5. **Don't commit secrets** - use .env references instead

## Integration with CLAUDE.md

The main CLAUDE.md file contains development guidelines and architecture.
This .claude/ directory contains operational context and quick references.

- CLAUDE.md = How to develop
- .claude/ = How to operate and troubleshoot