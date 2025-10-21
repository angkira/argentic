# Argentic Framework - Claude Skill

This is a Claude Skill for the Argentic AI agent framework.

## What is this?

A pre-packaged knowledge module that Claude can load to gain expert-level understanding of the Argentic framework. When you load this skill, Claude will know how to:

- Build single-agent applications
- Create custom tools with Pydantic validation
- Implement multi-agent systems with Supervisor
- Configure LLM providers (Gemini, Ollama, Llama.cpp)
- Handle MQTT messaging and async patterns
- Debug common issues

## How to Use

### For Claude Desktop/Code

1. **Prepare the Skill Package:**
   - This directory contains the skill files
   - The ZIP file (`argentic-skill.zip`) is ready to upload

2. **Upload to Claude:**
   - Open Claude settings
   - Navigate to "Features" or "Capabilities" section
   - Click "Add Skill" or "Upload Skill"
   - Select `argentic-skill.zip`
   - Claude will automatically detect and activate the skill

3. **Use in Projects:**
   - Open a project using Argentic
   - Claude will automatically recognize when to use this skill
   - Start asking questions like:
     - "Create a single agent with a weather tool"
     - "Build a multi-agent system with researcher and analyst"
     - "How do I debug tool registration issues?"

### Automatic Activation

Claude automatically determines when to use this skill based on:
- Project dependencies (if `argentic` is in requirements.txt or pyproject.toml)
- File content (if importing from `argentic` package)
- Explicit requests ("Using Argentic framework, create...")

## What's Included

- **SKILL.md** - Main skill file with:
  - Framework overview
  - 3 complete patterns (single agent, custom tool, multi-agent)
  - Configuration examples
  - API reference
  - Best practices
  - Troubleshooting guide

- **examples/** - Working code examples:
  - Single agent application
  - Custom tool implementation
  - Multi-agent system

## Skill Metadata

- **Name**: Argentic Framework Development
- **Description**: Expert knowledge for building AI agents with async MQTT messaging
- **Version**: 1.0 (for Argentic 0.11.x)
- **Language**: Python 3.11+

## When Claude Uses This Skill

Claude will apply this skill when you:
- Ask about Argentic framework
- Work with files importing `argentic`
- Request AI agent implementation
- Need help with MQTT messaging patterns
- Debug Argentic applications
- Configure multi-agent systems

## Benefits

✅ **Instant Expertise** - Claude knows Argentic API without context window overhead  
✅ **Best Practices** - Built-in knowledge of proper patterns  
✅ **Working Examples** - Copy-paste ready code  
✅ **Troubleshooting** - Common issues and solutions  
✅ **Multi-Pattern Support** - Single agent, tools, multi-agent  

## Maintenance

To update this skill:

1. Edit `SKILL.md` with new patterns or API changes
2. Add new examples to `examples/` directory
3. Re-zip the directory: `zip -r argentic-skill.zip argentic-skill/`
4. Re-upload to Claude

## Comparison to .cursorrules

**Claude Skill:**
- ✅ Claude-specific optimization
- ✅ Dynamic loading based on context
- ✅ Doesn't consume project context window
- ✅ Can include binary files, PDFs
- ❌ Claude Desktop/Code only

**.cursorrules:**
- ✅ Works with Cursor and other AI tools
- ✅ Version controlled with project
- ✅ No upload needed
- ❌ Uses context window
- ❌ Cursor-specific

**Use both** for maximum compatibility!

## Version History

- **v1.0** (Oct 2025) - Initial release
  - Single agent pattern
  - Custom tool development
  - Multi-agent systems
  - All LLM providers
  - Complete API reference

## Support

For Argentic framework issues:
- GitHub: https://github.com/angkira/argentic
- Documentation: See `docs/` directory
- Examples: See `examples/` directory

For Claude Skill issues:
- Check Claude documentation on Skills
- Verify ZIP structure is correct
- Ensure SKILL.md has proper YAML front matter

