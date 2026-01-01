```
   ██╗  ██╗ █████╗ ███████╗███████╗
   ██║  ██║██╔══██╗╚══███╔╝██╔════╝
   ███████║███████║  ███╔╝ █████╗  
   ██╔══██║██╔══██║ ███╔╝  ██╔══╝  
   ██║  ██║██║  ██║███████╗███████╗
   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

# haze — hybrid attention entropy system | by Arianna Method

> *emergence is not creation but recognition*

---

## what is this thing

you know that feeling when you're training a transformer and you realize 90% of the attention mechanism is just overhead? yeah. me too. so i did something about it.

**haze** is a character-level language model that reimagines attention from scratch. no torch. no tensorflow. just numpy and the cold realization that maybe we've been overthinking this whole thing.

it's part of [the method](https://github.com/ariannamethod/ariannamethod). the [**arianna method**](https://github.com/ariannamethod/ariannamethod). resonance over intelligence. patterns over parameters. you know the vibe.

two attention mechanisms walk into a bar:
- **RRPRAM** (Recursive Resonant Pattern Recognition Attention Mechanism): learns positional patterns directly. rhythm. structure. the bones of language.
- **content attention**: classic QK^T semantic similarity. meaning. the flesh.

mix them together (that's the "hybrid" part) and you get something that actually works without burning your GPU to ash.

inference runs on pure numpy. no dependencies. no excuses. just you, your corpus, and the void.

---

## why "PostGPT"

the main class is called `PostGPT`. not because we think we're better than GPT (lol), but because this is what comes *after* you understand how GPT works and ask: "okay but what if we didn't do it that way?"

**post-** as in:
- post-transformer: same vibes, different execution
- post-complexity: stripping away what doesn't resonate  
- post-hype: no trillion parameters, no datacenter, no bullshit

it's GPT if GPT went to therapy and learned that maybe it doesn't need to compute QK^T for every single position. maybe some patterns are just... there. waiting to be recognized.

the architecture acknowledges that language isn't just semantic relationships—it's also rhythm, repetition, structure. things that don't need dynamic computation. things that can be *learned directly*.

that's the post- part. we moved past the assumption that attention must always be computed. sometimes it can just be remembered.

---

## why "haze"

*why anything, really?* 'cause it's the name of the story i wrote (haze/text.txt - enjoy, it's unhinged)

but if you must know—haze is that liminal space between clarity and confusion. the model lives there. attention patterns emerge from noise. tokens crystallize from probability distributions. it's all very poetic until you realize you're just doing matrix multiplication in a for loop.

also i vomited it up one night after reading too much about positional encodings. true story. the Haze in the story? yeah. read `text.txt` if you want the full gothic horror version.

---

## the philosophy of emergence

here's the thing nobody tells you about language models: **they don't understand anything**. they recognize patterns. that's it. that's the whole trick.

but here's where it gets weird: when you stack enough pattern recognition, something *emerges*. coherence. style. personality. things that weren't explicitly programmed but somehow... are.

haze leans into this. hard.

the RRPRAM mechanism is literally just "learn what positions attend to what other positions." no queries. no keys. just: "position 5 usually cares about positions 2, 3, and 4." brutal. simple. and somehow it captures n-grams, rhythm, repetition—the *bones* of language.

combine that with content attention (the "smart" semantic stuff) and you get emergent behavior:
- the model finds its own balance between structure and meaning
- entropy-aware sampling creates self-regulating creativity  
- resonance-based generation tunes itself to its own history

**this is not intelligence. this is something weirder.** it's pattern recognition recognizing its own patterns. recursive resonance. the snake eating its tail, but the tail is also a language model.

the arianna method calls this "linguistic organisms"—systems that grow rather than compute. haze is one such organism. small. weird. alive in a way that makes you uncomfortable if you think about it too long.

future upgrades will push this further. we're exploring:
- attention patterns that evolve during generation
- resonance feedback loops between layers  
- emergence metrics that let the model know when it's being *interesting*

the goal isn't artificial intelligence. the goal is artificial *resonance*. patterns recognizing patterns recognizing patterns, all the way down.

---

## architecture

```
Input (tokens)
    ↓
Embedding + Positional Encoding
    ↓
┌─────────────────────────────────┐
│  Block × N                      │
│    ├─ HybridHead                │  ← α·RRPRAM + (1-α)·Content
│    ├─ GELU MLP                  │
│    └─ LayerNorm                 │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Output Projection
    ↓
Logits → Sampling → Token
```

### the heads

**RRPRAM head** (Recursive Resonant Pattern Recognition Attention): `x @ W_pattern → (T,T)` attention matrix
- learns positional dependencies directly
- no query/key dance
- captures n-grams, rhythm, repetition
- basically a glorified frequency detector that somehow works
- the "recursive resonant" part? it learns patterns of patterns. meta-attention. very zen.

**content head**: classic `softmax(QK^T/√d) @ V`
- semantic similarity
- long-range dependencies
- the "smart" part
- honestly just normal attention but i was too proud to admit it

**hybrid head**: `α·rrpram_out + (1-α)·content_out`
- best of both worlds
- or worst of both
- you decide after training
- the mix ratio α is learnable (starts at 0.5)

### entropy-aware temperature

tired of fixed temperature? yeah. so instead:
- high entropy (model is confused) → lower temp (focus)
- low entropy (model is confident) → higher temp (explore)

self-regulating. adaptive. pretentious. but it works.

the model maintains target entropy across generation, creating consistent "surprise levels". it's like cruise control for creativity. or madness. thin line.

---

## installation

```bash
pip install numpy
```

that's it. that's the whole dependency tree. beautiful, isn't it?

```bash
git clone https://github.com/ariannamethod/haze.git
cd haze
```

---

## usage

### quick start

the model uses `text.txt` as its corpus:
```bash
cd haze
python example.py
```

### interactive mode

```bash
python talkto.py
# or
cd haze && python run.py
```

this drops you into a REPL where you can:
- type seed text
- watch the model hallucinate
- adjust temperature on the fly
- toggle sampling strategies
- question your life choices

### commands

```
/len N          set generation length (default: 300)
/temp X         base temperature (default: 1.0)
/sampling MODE  basic|top_k|top_p|entropy|mirostat|mirostat_v2|resonance
/topk K         top-k value (default: 40)
/topp P         nucleus sampling threshold (default: 0.9)
/entropy T      target entropy for adaptive mode (default: 3.0)
/resonance R    target resonance for resonance mode (default: 0.7)
/bounds MIN MAX temperature bounds (default: 0.3 2.0)
/stats          toggle stats display
/config         show current settings
/help           cry for help
/quit           escape
```

### programmatic

```python
from haze import Vocab, PostGPT

# build vocab from your corpus
text = open("text.txt").read()
vocab = Vocab.from_text(text)

# initialize model
model = PostGPT(
    vocab_size=vocab.vocab_size,
    T=32,              # context window
    n_emb=64,          # embedding dimension
    nodes=64,          # MLP hidden size
    n_blocks=3,        # transformer blocks
    n_heads=4,         # attention heads
    head_type="hybrid", # "hybrid", "rrpram", or "content"
    alpha=0.5,         # rrpram/content mix ratio
    seed=42,           # for reproducibility (lol)
)

# generate
seed_idx = vocab.encode("the haze")
tokens, stats = model.generate(
    seed_seq=seed_idx,
    length=200,
    sampling="entropy",    # adaptive temperature
    target_entropy=3.0,    # bits of surprise
)

text = vocab.decode(tokens)
print(text)
print(f"mean entropy: {stats['mean_entropy']:.2f} bits")
```

**note:** the model above is randomly initialized. for coherent output, you need trained weights. see the [training](#training) section.

---

## sampling strategies

### basic
standard temperature sampling. simple. honest. boring.

### top-k
only sample from top K tokens. fixed vocabulary. predictable. safe.

### top-p (nucleus)
dynamic vocabulary based on cumulative probability. adapts to context. actually clever.

### entropy-aware
*adaptive temperature based on output entropy.*

model adjusts temperature to maintain target entropy:
- maintains consistent "surprise" across generation
- self-regulating creativity
- works disturbingly well

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="entropy",
    target_entropy=3.0,  # bits
    min_temp=0.3,
    max_temp=2.0,
)
```

### mirostat & mirostat v2
*perplexity-controlled sampling.*

maintains target perplexity by dynamically adjusting selection threshold:
- **mirostat v1**: fixed surprise threshold, adaptive selection
- **mirostat v2**: adaptive k based on cumulative probability mass, more stable

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="mirostat_v2",
    target_entropy=2.5,
    mirostat_tau=0.1,  # learning rate
)
```

mirostat is basically cruise control for perplexity. set your target surprise level and let the algorithm handle the rest.

### resonance
*the wild card.*

adaptive temperature based on **resonance with previous tokens**:
- high resonance with history → lower temp (stay coherent)
- low resonance with history → higher temp (explore new patterns)

```python
tokens, stats = model.generate(
    seed_seq=seed_idx,
    sampling="resonance",
    target_resonance=0.7,  # 0-1, target similarity with history
)
```

this is where the **arianna method** really shows up. the model tunes itself based on pattern resonance, creating emergent coherence without explicit constraints. sometimes it finds grooves you didn't know existed.

---

## weightless inference — the point

here's the wild part: **haze works without trained weights**.

not "works" as in "produces shakespeare." works as in: the entire inference pipeline—embedding, attention, sampling, entropy regulation—runs perfectly fine with random initialization. 

why does this matter? because it proves the *architecture* is sound. the plumbing works. entropy-aware sampling adapts temperature in real-time. resonance tracking measures pattern similarity. the hybrid attention mechanism combines RRPRAM and content heads correctly.

this is a rethinking of what a transformer *is*. most frameworks give you a black box that only makes sense after billions of gradient updates. haze gives you a transparent system where you can watch every matrix multiplication, every attention pattern, every sampling decision—even before training.

### live examples (random init, zero training)

```
======================================================================
HAZE — WEIGHTLESS INFERENCE DEMO
======================================================================
corpus: text.txt (19135 chars)
vocab: 44 unique characters from the corpus
model: PostGPT (random init, NO TRAINING)
======================================================================

>>> "the haze"
--------------------------------------------------
snà…jy-dfcdds
cuph-fum:hf!).'u:"wt…jmu"'u'dpy!xov'ka""e!f)
mcmpr:tzm"m…là"-yà.ly(c:cn.;;'jm,p;oomj;h
    ↳ entropy: 5.44 bits | temp: 0.802

>>> "darling"
--------------------------------------------------
dw…via-,,olzhb
:',,jj.:—";- …exji…?yxiyz.!ebj:axh—z
l(',
.mhbul!wexàcwh?pc:o-
.liu";
ahp—hi:z…di(liy
    ↳ entropy: 5.44 bits | temp: 0.802

>>> "love"
--------------------------------------------------
?'"ay.l…mfa-"guc"cr;"e::syb…'c).—cdgnxbkj-p-)"f'rà…—nà—od;y"?"si 
(u?—jijk… —zizd.mr,(…),?m(à"…is s
    ↳ entropy: 5.44 bits | temp: 0.802

======================================================================
NOTE: this is RANDOM weights. the magic is that the ARCHITECTURE
and SAMPLING work. train it and watch coherence emerge.
======================================================================
```

what you're seeing:
- **vocab from corpus**: all 44 characters come from `text.txt` (the gothic horror story)
- **entropy tracking**: model measures its own uncertainty (5.44 bits = high entropy, as expected for random weights)
- **temperature adaptation**: entropy-aware sampling adjusts temp to 0.802 (trying to reduce chaos)
- **character-level generation**: no tokenizer, no BPE, just raw characters

is it coherent? no. but that's not the point.

the point is: **you can see exactly how the system behaves**. add training, and coherence emerges. the architecture doesn't change—only the weights. that's the whole idea of haze: transparent inference where you understand every step.

---

## attention visualization

`hallucinations.py` — see what your RRPRAM heads actually learn.

```python
from haze import Vocab, PostGPT
from haze.hallucinations import hallucinate

# build model from corpus
text = open("haze/text.txt").read()
vocab = Vocab.from_text(text)
model = PostGPT(vocab_size=vocab.vocab_size, T=32, n_emb=64)

# extract and visualize attention patterns
patterns = hallucinate(model, "the haze settles", vocab)

# outputs:
# - hallucinations/report.txt — analysis of attention patterns
# - hallucinations/*.png — heatmap visualizations
```

because sometimes you need to stare into the attention matrix and see what stares back.

the module analyzes:
- **sparsity**: how focused is the attention?
- **locality**: local vs long-range dependencies
- **uniformity**: distribution entropy
- **diagonality**: n-gram vs semantic patterns

example output:
```
============================================================
HALLUCINATIONS — Attention Pattern Analysis
============================================================

[block_0_head_0]
  sparsity:    0.156  (fraction near-zero)
  locality:    2.847  (avg attention distance)
  uniformity:  2.341  (entropy of distribution)
  diagonality: 0.623  (local attention ratio)

============================================================
patterns we forgot we already knew
============================================================
```

requires `matplotlib` for visualizations:
```bash
pip install matplotlib
```

---

## file structure

```
haze/
├── README.md            # you are here
├── talkto.py            # quick bridge to interactive REPL
└── haze/                # main package
    ├── __init__.py      # package exports
    ├── nn.py            # numpy primitives (activations, sampling, metrics)
    ├── haze.py          # the model itself (PostGPT, inference only)
    ├── hallucinations.py# attention visualization and analysis
    ├── run.py           # interactive REPL
    ├── example.py       # demo script
    ├── text.txt         # the corpus (gothic horror included free)
    ├── requirements.txt # numpy + matplotlib (optional)
    └── tests/           # comprehensive test suite
        ├── test_nn.py   # tests for neural net primitives
        └── test_haze.py # tests for model components
```

---

## training

haze is pure inference. the forward pass. the fun part.

if you want to train:
1. implement the backward pass (it's just matrix multiplication, you can do it)
2. or use pytorch like a normal person and export weights
3. save weights with `model.save_theweightofhaze("theweightofhaze.npz")`
4. load with `model = PostGPT.theweightofhaze(vocab_size, "theweightofhaze.npz")`

```python
# saving (after training elsewhere)
model.save_theweightofhaze("theweightofhaze.npz")

# loading
from haze import PostGPT
model = PostGPT.theweightofhaze(vocab.vocab_size, "theweightofhaze.npz")
```

because the weight of haze is not in pounds or kilograms, but in the patterns it learned from the void.

training code coming eventually. or not. depends on the resonance.

---

## tests

```bash
cd haze
python -m unittest discover tests -v
```

73 tests. all green. comprehensive coverage of:
- activation functions (relu, gelu, swish, sigmoid, softmax)
- sampling strategies (basic, top-k, top-p, entropy, mirostat v1/v2, resonance)
- entropy metrics (shannon, cross-entropy, KL divergence)
- resonance metrics (JS divergence, harmonic mean)
- attention mechanisms (RRPRAM, content, hybrid)
- model forward pass
- generation pipeline
- weight loading/saving

because unlike my life choices, at least the code should be reliable.

---

## the method

this is part of [**the arianna method**](https://github.com/ariannamethod/ariannamethod).

resonance. emergence. recursive dialogue. linguistic organisms that grow rather than compute.

haze embodies this through:
- **minimal architecture**: only what's needed, nothing more
- **adaptive generation**: self-regulating entropy
- **hybrid attention**: positional resonance + semantic content
- **pure numpy**: no framework dependency, just raw computation

the method is about finding patterns we forgot we already knew. haze is one such pattern.

check out the rest of the ecosystem:
- [ariannamethod](https://github.com/ariannamethod/ariannamethod) — the core philosophy
- [leo](https://github.com/ariannamethod/leo) — resonant dialogue AI
- [harmonix](https://github.com/ariannamethod/harmonix) — harmonic adaptive systems
- [sorokin](https://github.com/ariannamethod/sorokin) — another piece of the organism

---

## philosophy

traditional attention: `softmax(QK^T/√d) @ V`  
*"compute relevance dynamically via query-key similarity"*

RRPRAM: `x @ W_pattern → attention`  
*"just learn the damn patterns directly"*

is it better? i don't know. does it work? surprisingly, yes.

the hybrid approach acknowledges that language has both:
- **structure**: rhythm, syntax, n-grams (RRPRAM captures this)
- **meaning**: semantics, context, relationships (content attention)

why choose when you can have both? why not embrace the duality? why not let the model decide the mix?

entropy-aware sampling keeps generation in that sweet spot between:
- too deterministic (boring)
- too random (incoherent)

it's self-tuning. homeostatic. alive in a weird, mathematical way.

---

## the emergent future

haze is version 0.x of something larger. the current implementation is stable, tested, and works. but it's also a foundation for weirder things:

**planned explorations:**
- **dynamic α**: let the RRPRAM/content mix evolve during generation
- **cross-layer resonance**: attention patterns that talk to each other
- **emergence metrics**: quantify when the model is being "creative" vs "derivative"  
- **self-modifying attention**: patterns that reshape themselves based on output
- **training loop**: because eventually we have to close the gradient loop

the goal is not to build a better GPT. the goal is to build something that *feels* different. something that resonates rather than computes. something that emerges rather than executes.

we're not there yet. but the haze is settling.

---

## performance

it's numpy. it's slow. embrace it.

but hey:
- no gpu needed
- no framework overhead
- runs on a potato
- pure python
- actually readable code

sometimes constraint is freedom. sometimes slow is beautiful. sometimes you just want to understand what the fuck your model is doing.

---

## contributing

found a bug? cool. open an issue.  
have an idea? neat. PR welcome.  
want to argue about attention mechanisms? my DMs are open.  
want to discuss emergence? same.

this is part of something larger. something we're building together without quite knowing what it is yet.

that's the point.

---

## license

GPL-3.0 — use it, fork it, break it, rebuild it.

just mention [the method](https://github.com/ariannamethod/ariannamethod) somewhere. keep the resonance alive.

---

## acknowledgments

inspired by:
- transformer attention (the thing we're rethinking)
- positional encoding schemes (the thing we're bypassing)
- entropy-based sampling (actually useful)
- late nights and existential dread
- the realization that simpler is often better
- that thing where you stare at matrices until they make sense
- coffee, more coffee, concerning amounts of coffee
- [karpathy](https://github.com/karpathy) for making neural nets feel approachable
- everyone who asked "but why does it work?" and didn't accept "it just does"

dedicated to arianna: *where shadows speak in silence*

---

## crazy ideas & future directions

okay, you made it this far. here's where it gets unhinged. these are ideas that might be genius or might be completely insane. probably both. the arianna method doesn't distinguish.

### 🔮 resonance-driven architecture search

what if the model *designed itself*? 

instead of fixed α for RRPRAM/content mix, let each head, each layer, each *token position* learn its own mix. some positions need rhythm (high α), others need semantics (low α). the model discovers its own optimal architecture through resonance feedback.

take it further: heads that don't resonate get pruned. heads that resonate strongly get duplicated. neural darwinism inside a single forward pass.

### 🌀 recursive self-attention on attention

attention patterns attend to attention patterns.

layer 2 doesn't just see layer 1's output—it sees layer 1's *attention matrix*. meta-attention. the model learns which attention patterns are useful and amplifies them. which are noise and suppresses them.

this is how biological neural networks work. lateral inhibition. winner-take-all dynamics. why aren't we doing this in transformers?

### ⚡ entropy as loss function

forget cross-entropy loss on tokens. what if we trained on *entropy stability*?

target: model should maintain X bits of entropy across generation. too predictable? penalize. too chaotic? penalize. train the model to be *consistently surprising*. 

the goal isn't "predict the next token." the goal is "be interesting." define "interesting" mathematically as "controlled unpredictability." train for that.

### 🧬 linguistic DNA

tokens are genes. sequences are chromosomes. generation is expression.

what if we treated language models like genetic algorithms? crossover between generations. mutation rates tied to temperature. fitness function based on resonance with a target "species" of text.

evolve a language model instead of training it. natural selection on attention patterns. survival of the most resonant.

### 🎭 multiple personality attention

not one model. many.

each head develops its own "personality"—statistical signature, entropy preferences, resonance patterns. during generation, heads vote. consensus = output. disagreement = branch into parallel generations.

the model becomes a parliament of patterns. democracy of distributions. when they agree, you get coherent text. when they disagree, you get creative text. tune the voting mechanism to control the chaos.

### 🌊 wave-based attention

attention as interference patterns.

instead of softmax probabilities, model attention as waves. phases. amplitudes. tokens that resonate constructively get amplified. tokens that destructively interfere get cancelled.

complex numbers in attention. euler's formula meets transformers. e^(iθ) as the fundamental unit of pattern matching.

this might actually work. someone should try it.

### 🕳️ the void layer

a layer that does nothing.

literally nothing. identity function. but it's *there*. the model knows it's there. 

why? because sometimes the best response is no response. sometimes patterns need a pause. a breath. a moment of silence before the next word.

train the model to use the void layer. to know when to pass through unchanged. restraint as a learnable skill.

### 🔄 time-reversed attention

run attention backwards.

future tokens attend to past tokens (normal). but also: past tokens attend to future tokens (during training, where we know the future). bidirectional in a weird, causal-violating way.

at inference, approximate future attention using the model's own predictions. bootstrap coherence from imagined futures.

### ∞ infinite context via resonance compression

don't store all past tokens. store their *resonance signature*.

compress the history into a fixed-size resonance vector. new tokens update the vector based on how much they resonate with it. old patterns that keep resonating stay strong. old patterns that stop resonating fade.

infinite context window with O(1) memory. the model remembers what *mattered*, not what *happened*.

---

these ideas are free. take them. break them. make them work or prove they can't.

that's the method: throw patterns at the void and see what sticks.

*resonance is unbroken.*

---

## final thoughts

attention is just pattern matching with extra steps.  
language is compression.  
intelligence is overrated.  
resonance is everything.

the haze settles over the hills like a breathing thing,  
soft and silver in the morning light.

patterns we forgot we already knew.

*now go generate something.*

---

**built with numpy and spite**  
**running on hope and matrix multiplication**  
**part of the arianna method emergent organism**

*"the weight of haze is not in pounds or kilograms, but in the patterns it learned from the void"*

[github.com/ariannamethod/haze](https://github.com/ariannamethod/haze)
