#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Entry point for the Rock Core Lamina Identification System."""

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from rock_core_analyzer.gui import main
    main()
