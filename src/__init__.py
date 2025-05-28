"""
Saorsa: Voice-Controlled SO-101 Robot Arms System

A natural language interface for controlling robot arms using Mac M3 hardware.
"""

__version__ = "0.1.0"
__author__ = "David Irvine"
__email__ = "david.irvine@maidsafe.net"
__license__ = "Apache 2.0"

from .main_mac import main

__all__ = ["main"]