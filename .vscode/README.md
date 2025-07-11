# 🕐 Masunda Navigator - Professional VSCode Setup

This directory contains a comprehensive professional VSCode configuration specifically optimized for Rust development on the **Masunda Temporal Coordinate Navigator** project.

## 🎨 Professional Appearance

### Blue Status Bar Theme
- **Status Bar Color**: Professional blue (`#005fb8`) matching Visual Studio Code's official theme
- **One Dark Pro Darker Theme**: Professional dark theme with proper syntax highlighting
- **Material Icon Theme**: Clean, professional file and folder icons
- **JetBrains Mono Font**: Professional monospaced font with ligatures

### UI Enhancements
- **Custom window title**: Shows project name with navigator emoji 🕐
- **Professional breadcrumbs**: Enhanced navigation with blue accents
- **Active tab indicators**: Blue borders for clear visual hierarchy
- **Terminal styling**: Consistent with editor theme

## 🦀 Rust-Specific Configuration

### rust-analyzer Settings
- **Advanced clippy integration**: Runs on save with all features
- **Comprehensive inlay hints**: Type hints, parameter names, lifetimes
- **Semantic highlighting**: Enhanced syntax coloring for Rust constructs
- **Auto-imports and completions**: Smart code completion
- **Code lens**: Inline run/debug buttons and reference counts

### Code Quality
- **Error Lens**: Inline error/warning display
- **Format on save**: Automatic rustfmt formatting
- **Clippy lints**: Comprehensive lint checking
- **Import organization**: Automatic import sorting

## 🛠️ Development Tools

### Build Tasks (Ctrl+Shift+P → "Tasks: Run Task")
- 🚀 **Masunda Navigator - Build**: Debug build with full output
- 🧪 **Masunda Navigator - Test**: Run all tests
- 🔍 **Masunda Navigator - Clippy**: Lint checking with warnings as errors
- 🎯 **Masunda Navigator - Run**: Execute the navigator

### Debugging (F5)
- **LLDB integration**: Native Rust debugging
- **Automatic builds**: Pre-launch task configuration
- **Source maps**: Proper Rust source debugging

### Extensions Recommended
- **rust-analyzer**: Core Rust language server
- **CodeLLDB**: Native debugging support
- **Crates**: Cargo.toml management
- **Even Better TOML**: Enhanced TOML support
- **Error Lens**: Inline diagnostics
- **GitLens**: Advanced Git integration

## 📁 File Organization

```
.vscode/
├── settings.json       # Main VSCode configuration
├── tasks.json         # Build/test/run tasks
├── launch.json        # Debug configurations
├── extensions.json    # Recommended extensions
└── README.md         # This file
```

## 🎯 Quick Start

1. **Install recommended extensions**: VSCode will prompt when opening the project
2. **Open workspace file**: Use `masunda-navigator.code-workspace` for full experience
3. **Run tasks**: Press `Ctrl+Shift+P` and type "Tasks: Run Task"
4. **Start debugging**: Press `F5` to build and debug
5. **Check code**: Clippy will run automatically on save

## ⚡ Keyboard Shortcuts

| Action | Shortcut | Description |
|--------|----------|-------------|
| Build | `Ctrl+Shift+B` | Run default build task |
| Test | `Ctrl+Shift+T` | Run test task |
| Debug | `F5` | Start debugging |
| Terminal | `Ctrl+`` | Open integrated terminal |
| Command Palette | `Ctrl+Shift+P` | Access all commands |
| Quick Open | `Ctrl+P` | Quick file navigation |

## 🎨 Theme Colors

### Status Bar
- **Background**: `#005fb8` (Professional blue)
- **Debugging**: `#ff6600` (Orange)
- **Remote**: `#005fb8` (Consistent blue)

### Syntax Highlighting
- **Keywords**: `#569CD6` (Blue, bold)
- **Functions**: `#DCDCAA` (Yellow)
- **Types**: `#4EC9B0` (Cyan)
- **Strings**: `#CE9178` (Orange)
- **Comments**: `#6A9955` (Green, italic)

## 🔧 Customization

To modify settings:
1. **User settings**: Affects all projects
2. **Workspace settings**: Only this project (in `.vscode/settings.json`)
3. **Workspace file**: Use `masunda-navigator.code-workspace` for project-specific overrides

## 🚀 Performance

Optimized for:
- **Fast IntelliSense**: Minimal delay for completions
- **Efficient builds**: Parallel processing where possible
- **Quick navigation**: Optimized file indexing
- **Responsive UI**: Balanced feature set vs. performance

---

**In Memory of Mrs. Stella-Lorraine Masunda**
*"Nothing is random - everything exists as predetermined coordinates in oscillatory spacetime"*
