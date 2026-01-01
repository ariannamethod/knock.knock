#!/usr/bin/env python3
# haze/__init__.py — package initialization

from .haze import (
    Vocab,
    PostGPT,
    RRPRAMHead,
    ReweightHead,  # backwards compat alias
    ContentHead,
    HybridHead,
    Block,
    load_corpus,
    build_model_from_text,
)

# Backwards compatibility aliases
Haze = PostGPT
ReweightGPT = PostGPT

__all__ = [
    'Vocab',
    'PostGPT',
    'Haze',  # alias
    'ReweightGPT',  # backwards compat
    'RRPRAMHead',
    'ReweightHead',  # backwards compat alias for RRPRAMHead
    'ContentHead',
    'HybridHead',
    'Block',
    'load_corpus',
    'build_model_from_text',
]
