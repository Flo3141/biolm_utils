# Documentation Index

Welcome to BioLM 2.0 documentation! This guide helps you find what you need quickly.

## 📖 Documentation Structure

| Document | What's Inside | When to Read |
|----------|---------------|--------------|
| **[README.md](../README.md)** | Quick start, overview, basic usage | 👈 **START HERE** |
| **[INSTALLATION.md](INSTALLATION.md)** | Detailed setup, troubleshooting | Installing for first time |
| **[CONFIGURATION.md](CONFIGURATION.md)** | All parameters, examples | Configuring experiments |
| **[TESTING.md](TESTING.md)** | Test suite guide, running tests | Understanding/running tests |
| **[CI_CD.md](CI_CD.md)** | GitHub Actions workflows | Understanding CI/CD pipeline |
| **[PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)** | Create custom plugins | Building your own model |
| **[PUBLISHING.md](PUBLISHING.md)** | Release to PyPI | Publishing your plugin |

## 🚀 Quick Navigation

### I want to...

**...install BioLM**
→ Start with [README Quick Start](../README.md#-quick-start) → [INSTALLATION.md](INSTALLATION.md) if issues

**...run my first experiment**
→ [README Quick Start](../README.md#-quick-start) → Copy template → Run

**...configure an experiment**
→ [CONFIGURATION.md](CONFIGURATION.md) → Check essential parameters section

**...understand what parameters do**
→ [CONFIGURATION.md Complete Reference](CONFIGURATION.md#complete-parameter-reference)

**...fix installation problems**
→ [INSTALLATION.md Troubleshooting](INSTALLATION.md#troubleshooting)

**...create a custom plugin**
→ [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md) → Follow step-by-step guide

**...publish my plugin to PyPI**
→ [PUBLISHING.md](PUBLISHING.md)

**...understand the test suite**
→ [TESTING.md](TESTING.md) → See what each test does

**...understand GitHub Actions workflows**
→ [CI_CD.md](CI_CD.md) → Learn how automation works

## 📚 Learning Path

### Beginners

1. **Install:** Follow [README Quick Start](../README.md#-quick-start)
2. **Verify:** Run the verification commands
3. **First Run:** Copy template, edit config, run training
4. **Understand:** Read [CONFIGURATION.md basics](CONFIGURATION.md#essential-parameters)

### Intermediate Users

1. **Advanced Config:** Study [CONFIGURATION.md complete reference](CONFIGURATION.md#complete-parameter-reference)
2. **Different Modes:** Learn tokenize → pre-train → fine-tune → predict pipeline
3. **Experiment:** Try both Saluki and XLNet plugins
4. **Optimize:** Tune hyperparameters for your data

### Advanced Users

1. **Plugin Dev:** Read [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)
2. **Implement:** Create your custom model
3. **Test:** Write tests for your plugin
4. **Share:** Publish to PyPI using [PUBLISHING.md](PUBLISHING.md)

## 🆘 Common Questions

**Q: Which file do I edit for my experiment?**
A: Copy `biolm/examples/plugin_template/config.yaml` and edit it. See [CONFIGURATION.md](CONFIGURATION.md).

**Q: What's the difference between Saluki and XLNet?**
A: Saluki is for RNA (12K token sequences, no pre-training). XLNet is for proteins (512 tokens, requires pre-training). See [README Available Plugins](../README.md#available-plugins).

**Q: My plugin isn't found. What's wrong?**
A: Check [INSTALLATION.md Plugin Not Discovered](INSTALLATION.md#plugin-not-discovered).

**Q: How do I know what parameters are required?**
A: See [CONFIGURATION.md Essential Parameters](CONFIGURATION.md#essential-parameters).

**Q: Can I create my own model?**
A: Yes! See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md).

## 🔗 External Resources

- **Poetry Documentation:** https://python-poetry.org/docs/
- **Hydra Configuration:** https://hydra.cc/docs/intro/
- **PyTorch:** https://pytorch.org/docs/
- **Transformers:** https://huggingface.co/docs/transformers/

## 📝 Documentation Maintenance

This documentation structure is designed to be:
- **Clear:** One main entry point (README), specialized docs in docs/
- **Navigable:** Each doc has a clear purpose
- **Maintainable:** No redundancy, single source of truth

When updating docs:
1. **README.md** - Keep it concise, focus on quick start
2. **INSTALLATION.md** - Detailed setup, troubleshooting
3. **CONFIGURATION.md** - Complete parameter reference
4. **PLUGIN_DEVELOPMENT.md** - Plugin creation guide
5. **PUBLISHING.md** - Release process

## 📊 Documentation Map

```
BioLM 2.0/
│
├── README.md ⭐ START HERE
│   ├── Quick start (3 commands)
│   ├── Data format
│   ├── Basic usage
│   ├── Available plugins
│   └── Links to detailed docs
│
└── docs/
    ├── INSTALLATION.md
    │   ├── Prerequisites
    │   ├── Quick install
    │   ├── Manual install
    │   ├── Verification
    │   └── Troubleshooting
    │
    ├── CONFIGURATION.md
    │   ├── Quick start
    │   ├── Essential parameters
    │   ├── Complete reference
    │   ├── Examples
    │   └── Troubleshooting
    │
    ├── PLUGIN_DEVELOPMENT.md
    │   ├── Architecture overview
    │   ├── Step-by-step guide
    │   ├── Template walkthrough
    │   ├── Testing
    │   └── Best practices
    │
    └── PUBLISHING.md
        ├── PyPI workflow
        ├── Version management
        └── Automation
```

## 🤝 Contributing to Docs

Found a typo? Want to improve clarity? Contributions welcome!

1. Edit the relevant file in `docs/` or root `README.md`
2. Keep language clear and concise
3. Add examples where helpful
4. Update this index if adding new docs
5. Submit a PR

---

**Need help?** Open an issue: https://github.com/dieterich-lab/biolm_utils/issues
