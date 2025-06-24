# One-File One-Command Projects

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE-AGPL)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A collection of minimal, single-file scripts and micro-projects I wrote myself, each living in its own language-specific folder under `src/` and following a strict "one file, one command" philosophy.

---

## 🚀 Scripts Index

TODO: Add scripts

---

## 🤝 Contributing

1. **Add your script**
   Place your standalone file in the appropriate folder under `src/`.
2. **Document it**
   Add a row to the **Scripts Index** table above (name, language, one-liner, usage).
3. **Follow style**

   * Use a shebang (`#!/usr/bin/env ...`) for interpreted scripts.
   * Highly desirable to implement `-h|--help` flag for usage instructions.
   * Stick to standard library (zero external deps).
4. **Submit a PR**
   CI will lint, test, and smoke-check your script automatically.

---

## 📜 License

This project is licensed under the [AGPL-3.0 License](LICENSE-AGPL).
By contributing, you agree to license your work under the same terms.
