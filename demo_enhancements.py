#!/usr/bin/env python3
"""
demo_enhancements.py — Demonstrate Enhanced Generation Quality

Shows the improvements in:
- Loop detection and avoidance
- Enhanced entropy-aware sampling
- Field enrichment with adaptive temperature
- Better coherence tracking
"""

import sys
from pathlib import Path

# Add haze to path
sys.path.insert(0, str(Path(__file__).parent))

from haze import Vocab, PostGPT
from haze.subword_field import SubwordField


def demo_loop_detection():
    """Demonstrate loop detection and avoidance."""
    print("=" * 70)
    print("DEMO 1: Loop Detection and Avoidance")
    print("=" * 70)
    print()
    
    # Create a simple vocab
    text = "the cat sat on the mat. the dog ran in the park. the bird flew in the sky."
    vocab = Vocab.from_text(text)
    
    # Create model
    model = PostGPT(
        vocab_size=vocab.vocab_size,
        T=16,
        n_emb=32,
        nodes=32,
        n_blocks=2,
        n_heads=2,
        seed=42,
    )
    
    seed_seq = vocab.encode("the cat")
    
    print("Standard entropy sampling:")
    tokens1, stats1 = model.generate(
        seed_seq,
        length=50,
        sampling="entropy",
        enable_loop_detection=False,  # Disabled
        target_entropy=2.0,
    )
    print(f"  {vocab.decode(tokens1)}")
    print(f"  Stats: entropy={stats1['mean_entropy']:.2f}, loops={stats1.get('loop_detections', 0)}")
    print()
    
    print("Loop-aware sampling:")
    tokens2, stats2 = model.generate(
        seed_seq,
        length=50,
        sampling="loop_aware",  # New mode!
        target_entropy=2.0,
    )
    print(f"  {vocab.decode(tokens2)}")
    print(f"  Stats: entropy={stats2['mean_entropy']:.2f}, loops={stats2.get('loop_detections', 0)}")
    print()


def demo_enhanced_entropy():
    """Demonstrate enhanced entropy sampling with momentum."""
    print("=" * 70)
    print("DEMO 2: Enhanced Entropy Sampling (v2)")
    print("=" * 70)
    print()
    
    text = "the cat sat on the mat. the dog ran in the park. the bird flew in the sky."
    vocab = Vocab.from_text(text)
    
    model = PostGPT(
        vocab_size=vocab.vocab_size,
        T=16,
        n_emb=32,
        nodes=32,
        n_blocks=2,
        n_heads=2,
        seed=43,
    )
    
    seed_seq = vocab.encode("the cat")
    
    print("Standard entropy sampling:")
    tokens1, stats1 = model.generate(
        seed_seq,
        length=40,
        sampling="entropy",
        target_entropy=2.0,
    )
    print(f"  Entropy std: {stats1['entropy_std']:.3f}")
    print()
    
    print("Enhanced entropy v2 (with momentum):")
    tokens2, stats2 = model.generate(
        seed_seq,
        length=40,
        sampling="entropy_v2",  # New mode!
        target_entropy=2.0,
    )
    print(f"  Entropy std: {stats2['entropy_std']:.3f} (should be lower - more stable)")
    print()


def demo_subword_enhanced():
    """Demonstrate enhanced subword generation with loop avoidance."""
    print("=" * 70)
    print("DEMO 3: Enhanced Subword Field Generation")
    print("=" * 70)
    print()
    
    # Check if sentencepiece is available
    try:
        import sentencepiece
    except ImportError:
        print("Sentencepiece not available. Skipping this demo.")
        print("Install with: pip install sentencepiece")
        return
    
    # Use actual corpus if available
    corpus_path = Path(__file__).parent / "haze" / "text.txt"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        print("Skipping this demo.")
        return
    
    print("Building subword field from corpus...")
    field = SubwordField.from_corpus(str(corpus_path), vocab_size=300)
    print(f"Field stats: {field.get_stats()}")
    print()
    
    seed = "I love"
    
    print(f"Standard generation from '{seed}':")
    result1 = field.generate(seed, length=30, temperature=0.75, mode="trigram")
    print(f"  {result1}")
    print()
    
    print(f"Enhanced generation (loop-aware, adaptive temp):")
    result2 = field.generate_enhanced(
        seed,
        length=30,
        temperature=0.75,
        mode="trigram",
        loop_penalty=0.4,
        adaptive_temp=True,
        target_entropy=2.5,
    )
    print(f"  {result2}")
    print()


def demo_coherence_tracking():
    """Demonstrate coherence tracking during generation."""
    print("=" * 70)
    print("DEMO 4: Coherence Tracking")
    print("=" * 70)
    print()
    
    text = "the cat sat on the mat. the dog ran in the park. the bird flew in the sky."
    vocab = Vocab.from_text(text)
    
    model = PostGPT(
        vocab_size=vocab.vocab_size,
        T=16,
        n_emb=32,
        nodes=32,
        n_blocks=2,
        n_heads=2,
        seed=44,
    )
    
    seed_seq = vocab.encode("the cat")
    
    print("Generating with coherence tracking...")
    tokens, stats = model.generate(
        seed_seq,
        length=40,
        sampling="entropy_v2",  # Modes that track coherence
        target_entropy=2.0,
    )
    
    print(f"Generated: {vocab.decode(tokens)[:100]}...")
    print()
    print("Coherence stats:")
    if "mean_coherence" in stats:
        print(f"  Mean coherence: {stats['mean_coherence']:.3f}")
        print(f"  Coherence std: {stats['coherence_std']:.3f}")
    else:
        print("  (Coherence tracking requires entropy_v2 or loop_aware mode)")
    print()
    
    print("All stats:")
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "HAZE ENHANCED GENERATION DEMO" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demos = [
        demo_loop_detection,
        demo_enhanced_entropy,
        demo_subword_enhanced,
        demo_coherence_tracking,
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            demo()
        except Exception as e:
            print(f"Error in demo {i}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos):
            print("\n")
    
    print("=" * 70)
    print("Demo complete!")
    print()
    print("Key improvements:")
    print("  ✓ Loop detection prevents repetitive patterns")
    print("  ✓ Enhanced entropy sampling has better stability")
    print("  ✓ Adaptive temperature adjusts to context")
    print("  ✓ Coherence tracking measures generation quality")
    print("=" * 70)


if __name__ == "__main__":
    main()
