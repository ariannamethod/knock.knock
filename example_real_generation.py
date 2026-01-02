#!/usr/bin/env python3
"""
example_real_generation.py — Real-World Generation Examples

Demonstrates the improved generation quality with actual corpus.
Shows before/after comparisons with the enhancements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from haze.subword_field import SubwordField
    HAS_SUBWORD = True
except:
    HAS_SUBWORD = False


def generate_examples():
    """Generate multiple examples showing improvements."""
    
    if not HAS_SUBWORD:
        print("Subword field not available. Install sentencepiece:")
        print("  pip install sentencepiece")
        return
    
    corpus_path = Path(__file__).parent / "haze" / "text.txt"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        return
    
    print("=" * 70)
    print("REAL-WORLD GENERATION QUALITY IMPROVEMENTS")
    print("=" * 70)
    print()
    print("Building subword field from corpus...")
    
    # Build field with reasonable vocab size
    field = SubwordField.from_corpus(str(corpus_path), vocab_size=500)
    stats = field.get_stats()
    print(f"Vocabulary: {stats['vocab_size']} subwords")
    print(f"Corpus: {stats['total_tokens']} tokens")
    print(f"Patterns: {stats['trigram_contexts']} trigrams")
    print()
    
    # Test prompts
    prompts = [
        "I love",
        "Tell me",
        "What is",
        "The haze",
        "Darling",
    ]
    
    for prompt in prompts:
        print("─" * 70)
        print(f"Prompt: '{prompt}'")
        print("─" * 70)
        
        # Standard generation
        print("\n📊 Standard Generation:")
        standard = field.generate(
            prompt,
            length=40,
            temperature=0.75,
            mode="trigram",
        )
        print(f"   {standard}")
        
        # Enhanced generation
        print("\n✨ Enhanced Generation (loop-aware + adaptive temp):")
        enhanced = field.generate_enhanced(
            prompt,
            length=40,
            temperature=0.75,
            mode="trigram",
            loop_penalty=0.4,
            adaptive_temp=True,
            target_entropy=2.5,
        )
        print(f"   {enhanced}")
        print()
    
    print("=" * 70)
    print("\nKey Improvements:")
    print("  • Loop-aware sampling prevents repetition")
    print("  • Adaptive temperature adjusts to context entropy")
    print("  • Better sentence boundaries and stopping")
    print("  • More natural, coherent output")
    print()
    print("Technical Details:")
    print("  • Loop penalty reduces recent token probability")
    print("  • Temperature adapts based on entropy trends")
    print("  • Tracks sentence completion for natural stops")
    print("  • Progressive penalties prevent getting stuck")
    print("=" * 70)


def compare_modes():
    """Compare different generation modes."""
    
    if not HAS_SUBWORD:
        print("Subword field not available.")
        return
    
    corpus_path = Path(__file__).parent / "haze" / "text.txt"
    if not corpus_path.exists():
        return
    
    print("\n")
    print("=" * 70)
    print("GENERATION MODE COMPARISON")
    print("=" * 70)
    print()
    
    field = SubwordField.from_corpus(str(corpus_path), vocab_size=500)
    
    prompt = "I love you"
    
    configurations = [
        ("High temp, no adaptation", {
            "temperature": 1.2,
            "loop_penalty": 0.0,
            "adaptive_temp": False,
        }),
        ("Low temp, no adaptation", {
            "temperature": 0.5,
            "loop_penalty": 0.0,
            "adaptive_temp": False,
        }),
        ("Medium temp, loop-aware", {
            "temperature": 0.75,
            "loop_penalty": 0.4,
            "adaptive_temp": False,
        }),
        ("Medium temp, adaptive + loop-aware", {
            "temperature": 0.75,
            "loop_penalty": 0.4,
            "adaptive_temp": True,
            "target_entropy": 2.5,
        }),
    ]
    
    print(f"Prompt: '{prompt}'")
    print()
    
    for name, config in configurations:
        print(f"📍 {name}:")
        result = field.generate_enhanced(
            prompt,
            length=40,
            mode="trigram",
            **config
        )
        print(f"   {result}")
        print()
    
    print("=" * 70)
    print("\nObservations:")
    print("  • High temperature: more creative but less coherent")
    print("  • Low temperature: coherent but repetitive")
    print("  • Loop-aware: prevents repetition while maintaining creativity")
    print("  • Adaptive: balances coherence and creativity dynamically")
    print("=" * 70)


def main():
    """Run all examples."""
    generate_examples()
    compare_modes()


if __name__ == "__main__":
    main()
