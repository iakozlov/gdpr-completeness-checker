# Deolingo Server Installation Guide

This guide helps you install Deolingo on a server to fix the "deolingo command not found" error.

## Quick Local Installation (Recommended)

If you have the deolingo folder in your project (which you do), this is the fastest method:

```bash
# Navigate to your project directory
cd /path/to/dpa-completeness-checker

# Install the local deolingo package in editable mode
pip3 install -e ./deolingo/

# Verify installation
deolingo --version
```

This installs deolingo from the local folder and makes the `deolingo` command available system-wide. 