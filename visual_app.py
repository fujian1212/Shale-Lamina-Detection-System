#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rock-core lamina detection GUI (compatibility entry point; prefer rock_core_analyzer.gui)."""

from rock_core_analyzer.gui import RockCoreAnalyzerApp, main

__all__ = ["RockCoreAnalyzerApp", "main"]

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
