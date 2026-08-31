"""Canonical CLI package for somehand."""

from .main import main
from .parser import build_parser

__all__ = ["build_parser", "main"]
