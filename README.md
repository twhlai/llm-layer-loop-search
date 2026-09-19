# llm-layer-loop-search

llm-layer-loop-search is a toolkit for experimenting with looped transformer layers.  It was inspired by [David Noel Ng's RYS](https://dnhkng.github.io/posts/rys-ii/) approach, but has two key differences.  First, instead of looping blocks of layers, I allow only per-layer looping.  Second, I repeat only the attention sublayer and not the feed-forward network.

# Background

[Ng](https://dnhkng.github.io/posts/rys/) and [Chen et al.](https://arxiv.org/pdf/2605.23872) separately observed that they could take an LLM and improve its performance by selectively looping layers, without any additional modification or training.  Ng found that the choice of looped layers could yield significant improvements, serious degradation, or anything in between, so he built a toolkit that can systematically evaluate all possible single-block repeats.

[Jack Tao](https://lucidchips.com/#/writing/spectral-screening-which-layers-to-repeat) found cases where avoiding repeated feed-forward network passes improved performance, and he suggested that repeated FFN passes can negatively affect knowledge recall.  [Lin et al.](https://arxiv.org/pdf/2608.18230) proposed an architecture called MixerLoop, which loops a layer's attention sublayer, but applies the feed-forward network only once.

With all this in mind, I wanted to experiment with looping attention sublayers in existing LLMs, so I made this minimalist toolkit.

# How to use

My code is a heavily modified fork of [llm-circuit-finder](https://github.com/alainnothere/llm-circuit-finder), a toolkit that works with GGUFs for evaluating layer duplication configurations.  The toolkit requires **python** and **llama.cpp**.

I didn't test my code on any hybrid or MoE LLMs, but it ran fine on Mistral Nemo. :)

### Setup

```bash
pip install gguf requests tqdm
```

### Evaluate different looped layer configurations

To test every single looped layer configuration among layers {10, ..., 30}:
```bash
python sweep.py \
    --model /path/to/model.gguf \
    --llama-server /path/to/llama-server \
    --tmpdir /dev/shm/rys \
    --results results.jsonl \
    --candidates 10..30 \
    --skip-baseline \
    --port 8099 \
    --server-args --device CUDA0
```
(By default, a looped layer has 2 iterations of the attention sublayer.)

To test every possible pair of looped layers among layers {10, 12, 14, 16}, with each looped layer having 4 iterations:
```bash
python sweep.py \
    --model /path/to/model.gguf \
    --llama-server /path/to/llama-server \
    --tmpdir /dev/shm/rys \
    --results results.jsonl \
    --candidates 10,12,14,16 \
    --num-loops 2 \
    --repeat-factor 4 \
    --skip-baseline \
    --port 8099 \
    --server-args --device CUDA0
```

### Build a modified GGUF for testing

```bash
# Duplicate layers or blocks of layers by listing the layers in order
python layer_path.py model.gguf improved.gguf \
    -p "0..14,12,13,14,15..39" -v

# -14 means "use layer 14's attention sublayer but not its FFN"
python layer_path.py model.gguf improved.gguf \
    -p "0..13,-14,14..39" -v
```

### Check for model degradation

```bash
python extra_tests.py \
    --model /path/to/model.gguf \
    --llama-server /path/to/llama-server \
    --tmpdir /dev/shm/rys \
    --port 8099 \
    --bbh-limit 20 \
    --output out.jsonl \
    --server-args --device CUDA0
```

## Files

| File | What it does |
|------|-------------|
| `eq_probe.py` | Emotional intelligence probe (EQ-Bench style) |
| `extra_tests.py` | Separate script for running additional BBH and TinyMMLU tests |
| `gguf_surgery.py` | Low-level GGUF construction functions |
| `layer_path.py` | Builds GGUF using specified layer execution path |
| `ls_utils.py` | Functions for llama-server handling |
| `math_probe.py` | Hard arithmetic probe (Ng's partial-credit scoring) |
| `reasoning_probe.py` | BBH-derived causal/logical/navigation/math word problems |
| `sweep.py` | Main sweep harness — evaluates different looped layer configs |

## License

MIT
