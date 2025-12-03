---
layout: default
title: Installation
---

# Installation Guide

## Prerequisites

* Python 3.12 or higher
* Go 1.21 or higher (for CLI)

## Python Installation

### Via pip

```bash
pip install pulsarfitpy
```

### From Source

```bash
git clone https://github.com/jfk-astro/pulsarfitpy.git
cd pulsarfitpy
pip install -e .
```

## Go CLI Installation

### Build from Source

```bash
# Clone the repository
git clone https://github.com/jfk-astro/pulsarfitpy.git
cd pulsarfitpy

# Build the CLI
make build
```

The binary will be available at `bin/pulsar-cli.exe` (Windows) or `bin/pulsar-cli` (Unix).

