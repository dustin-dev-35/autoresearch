"""
Experiment: trained-checkpoint seed variance of D_eff (arXiv:2609.31098v1).

Run one arm.  One variable per run, carried in the environment:

    AXV_RUN_ID        run identifier, also the output directory name
    AXV_RUN_DIR       where run.json / train.log / metrics.json land
    AXV_SEED          the one variable for the seed series
    AXV_N_LAYER       depth L (only varied by the between-architecture arm)
    AXV_GAMMA         residual scaling gamma (1.0 = unmodified residual = baseline)
    AXV_MAX_ITERS     fixed iteration budget, never a wall-clock budget
    AXV_N_PASSAGES    passages pooled for the D_eff estimate
    AXV_TRAIN         1 = train to budget, 0 = random-weight control (no training)
    AXV_TF32          1 = allow TF32 matmuls (default), 0 = strict fp32

Everything else is fixed by the paper's own Appendix S17 configuration:

    n_layer = 12, n_embd = 384, n_head = 6, block_size = 256,
    batch_size = 64, lr = 1e-3 constant, max_iters = 5000, shakespeare_char

Why a fixed *iteration* budget rather than the harness wall-clock budget:
D_eff is a property of a checkpoint, not of throughput, and the paper's
published anchor for this configuration is stated at "iter 5000".  A
wall-clock budget would move the anchor.  This is a new series with its own
cohort and its own baseline, and it is NOT comparable to any val_bpb
autoresearch arm.  See run.json "series_deviations".

Series deviations from the upstream autoresearch harness, all deliberate and all
recorded in metrics.json:
  - fixed-iteration budget instead of fixed wall clock (above)
  - char-level shakespeare_char instead of climbmix-400b + 8192-BPE, because the
    paper's own S17 anchor is stated on that corpus (new data_fingerprint)
  - plain pre-norm GPT decoder instead of the harness's sliding-window /
    value-embedding / Muon variant, so the architecture is the one Appendix S17
    describes
  - results leave via stdout, because the Runpod MCP surface has no exec channel

Usage:
    python train.py
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from deff import d_eff_from_layers, reference_depth, signed_gap, update_norm_ratio

# ---------------------------------------------------------------------------
# Fixed configuration.  Only AXV_SEED, AXV_N_LAYER, AXV_GAMMA and AXV_TRAIN vary.
# ---------------------------------------------------------------------------

N_EMBD = 384
N_HEAD = 6
BLOCK_SIZE = 256
BATCH_SIZE = 64
BASE_LR = 1e-3

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.environ.get("AXV_DATA_PATH", "shakespeare_char_input.txt")

RUN_ID = os.environ["AXV_RUN_ID"]
RUN_DIR = os.environ["AXV_RUN_DIR"]
SEED = int(os.environ.get("AXV_SEED", "1337"))
N_LAYER = int(os.environ.get("AXV_N_LAYER", "12"))
GAMMA = float(os.environ.get("AXV_GAMMA", "1.0"))
MAX_ITERS = int(os.environ.get("AXV_MAX_ITERS", "5000"))
N_PASSAGES = int(os.environ.get("AXV_N_PASSAGES", "10000"))
TRAIN = int(os.environ.get("AXV_TRAIN", "1"))

os.makedirs(RUN_DIR, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# TF32 matmuls. A fixed compute-mode choice applied identically to every arm in
# the series, so it cannot confound a seed-to-seed comparison.  It does mean
# these numbers are not bit-comparable to a strict-fp32 run of the same code.
TF32 = os.environ.get("AXV_TF32", "1") == "1"
if TF32 and DEV == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Data: shakespeare_char, char level.
# ---------------------------------------------------------------------------

def load_corpus() -> Dict[str, object]:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"missing corpus at {DATA_PATH!r}; fetch {DATA_URL} first "
            "(the run script does this and records the sha256 as data_fingerprint)"
        )
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    n = len(ids)
    n_train = int(n * 0.9)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:],
        "vocab_size": len(chars),
        "stoi": stoi,
        "n_chars": n,
    }


def get_batch(split: torch.Tensor, block: int, batch: int, gen: torch.Generator):
    hi = split.numel() - block - 1
    if hi <= 0:
        raise ValueError("split is shorter than block_size + 1")
    ix = torch.randint(hi, (batch,), generator=gen)
    x = torch.stack([split[i : i + block] for i in ix])
    y = torch.stack([split[i + 1 : i + 1 + block] for i in ix])
    return x.to(DEV), y.to(DEV)


# ---------------------------------------------------------------------------
# Model: plain pre-norm GPT decoder with a scalable residual path.
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = BLOCK_SIZE
    vocab_size: int = 65
    n_layer: int = N_LAYER
    n_head: int = N_HEAD
    n_embd: int = N_EMBD


class SelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.hd = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """h_{l+1} = gamma * h_l + attn(LN(h_l)); h_{l+2} = gamma * h_{l+1} + mlp(LN(h_{l+1})).

    Appendix S17 Eq S3.  gamma = 1.0 is the unmodified residual stream and is
    the experiment's baseline arm.
    """

    def __init__(self, cfg: GPTConfig, gamma: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gamma * x + self.attn(self.ln1(x))
        x = self.gamma * x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, gamma: float):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, gamma) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # tied
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(self._trunk(idx)))

    def _trunk(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            x = blk(x)
        return x

    def hidden_states(self, idx: torch.Tensor) -> List[torch.Tensor]:
        """Return [h_0 (embedding), h_1, ..., h_L].  D_eff uses h_1..h_L only."""
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        out = [x]
        for blk in self.blocks:
            x = blk(x)
            out.append(x)
        return out


# ---------------------------------------------------------------------------
# D_eff measurement, on the paper's protocol.
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_d_eff(model: GPT, split: torch.Tensor, n_passages: int, batch: int) -> Dict[str, object]:
    model.eval()
    hi = split.numel() - BLOCK_SIZE - 1
    need = min(n_passages, hi)
    # Deterministic, evenly spaced passage starts: no sampling noise here, so the
    # only stochastic element in this measurement is the training seed.  The
    # paper's passage-bootstrap floor (S18, width <= 3e-4) is a resampling floor
    # and is deliberately not re-introduced here; this run measures the seed axis.
    starts = torch.linspace(0, hi - 1, steps=need).round().long()
    pooled: Optional[torch.Tensor] = None
    count = 0
    for i in range(0, need, batch):
        chunk = starts[i : i + batch]
        x = torch.stack([split[s : s + BLOCK_SIZE] for s in chunk]).to(DEV)
        hs = model.hidden_states(x)
        # mean-pool over valid non-padding positions.  Sequences are packed at
        # full length, so every position is valid, the mask is all ones, and this
        # is an exact mean over BLOCK_SIZE positions.
        per_layer = torch.stack([h.to(torch.float32).mean(dim=1) for h in hs], dim=0)
        pooled = per_layer if pooled is None else torch.cat([pooled, per_layer], dim=1)
        count += per_layer.shape[1]
    assert pooled is not None
    layers = [pooled[i].to(torch.float64) for i in range(1, pooled.shape[0])]
    res = d_eff_from_layers(layers, L=N_LAYER, keep_matrix=True)
    matrix = res.pop("cka_matrix")  # type: ignore[misc]
    res["norm_ratio_update_over_state"] = update_norm_ratio(layers)
    res["cka_matrix"] = matrix.tolist()  # type: ignore[union-attr]
    res["n_valid_positions_per_passage"] = BLOCK_SIZE
    res["n_passages_used"] = int(count)
    model.train()
    return res


@torch.no_grad()
def evaluate(model: GPT, split: torch.Tensor, iters: int, batch: int, gen: torch.Generator) -> float:
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(split, BLOCK_SIZE, batch, gen)
        loss = F.cross_entropy(model(x).view(-1, model.head.out_features), y.view(-1))
        tot += float(loss)
    model.train()
    return tot / iters


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gen = torch.Generator(device="cpu").manual_seed(SEED)

    data = load_corpus()
    cfg = GPTConfig(vocab_size=int(data["vocab_size"]))
    model = GPT(cfg, GAMMA).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"run_id:        {RUN_ID}")
    print(f"device:        {DEV}")
    print(f"seed:          {SEED}")
    print(f"n_layer:       {N_LAYER}")
    print(f"n_embd:        {N_EMBD}")
    print(f"n_head:        {N_HEAD}")
    print(f"block_size:    {BLOCK_SIZE}")
    print(f"batch_size:    {BATCH_SIZE}")
    print(f"gamma:         {GAMMA}")
    print(f"max_iters:     {MAX_ITERS}")
    print(f"n_passages:    {N_PASSAGES}")
    print(f"train:         {TRAIN}")
    print(f"tf32:          {TF32}")
    print(f"model_config:  {asdict(cfg)}")
    print(f"num_params:    {n_params}")
    print(f"vocab_size:    {data['vocab_size']}")
    print(f"corpus_chars:  {data['n_chars']}")
    print(f"F_L:           {reference_depth(N_LAYER):.6f}", flush=True)

    decay, nodecay = [], []
    for _pn, p in model.named_parameters():
        (decay if p.dim() >= 2 else nodecay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
        lr=BASE_LR,
        betas=(0.9, 0.95),
    )

    losses: List[float] = []
    t_train0 = time.time()
    if TRAIN:
        for step in range(1, MAX_ITERS + 1):
            x, y = get_batch(data["train"], BLOCK_SIZE, BATCH_SIZE, gen)  # type: ignore[arg-type]
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()  # lr constant at 1e-3, per Appendix S17
            opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step % 500 == 0 or step == 1:
                el = time.time() - t_train0
                print(
                    f"iter {step:05d}/{MAX_ITERS} | train_loss: {losses[-1]:.4f} "
                    f"| mean100: {sum(losses[-100:]) / min(100, len(losses)):.4f} "
                    f"| elapsed_s: {el:.0f} | ms_per_iter: {1000 * el / step:.1f}",
                    flush=True,
                )
    train_seconds = time.time() - t_train0

    val_gen = torch.Generator(device="cpu").manual_seed(SEED + 10_000)
    val_loss = evaluate(model, data["val"], iters=40, batch=32, gen=val_gen)  # type: ignore[arg-type]
    t_measure = time.time()
    deff = measure_d_eff(model, data["val"], N_PASSAGES, batch=128)  # type: ignore[arg-type]
    t_end = time.time()

    print("---")
    print(f"d_eff:               {deff['D_eff']:.6f}")
    print(f"d_eff_over_L:        {deff['D_eff_over_L']:.6f}")
    print(f"F_L:                 {deff['F_L']:.6f}")
    print(f"gap_to_F_L:          {deff['gap']:.6f}")
    print(f"rho_lag1:            {deff['rho_lag1']:.6f}")
    print(f"norm_ratio_f_over_h: {deff['norm_ratio_update_over_state']:.6f}")
    print(f"n_passages_used:     {deff['n_passages_used']}")
    print(f"val_loss:            {val_loss:.6f}")
    print(f"train_iters_done:    {len(losses)}")
    print(f"train_seconds:       {train_seconds:.1f}")
    print(f"total_seconds:       {t_end - t_start:.1f}")
    if torch.cuda.is_available():
        print(f"peak_vram_mb:        {torch.cuda.max_memory_allocated() / 2**20:.1f}")
        print(f"device_name:         {torch.cuda.get_device_name(0)}")
        print(f"cuda_version:        {torch.version.cuda}")

    metrics = {
        "run_id": RUN_ID,
        "seed": SEED,
        "n_layer": N_LAYER,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "max_iters": MAX_ITERS,
        "trained": bool(TRAIN),
        "num_params": n_params,
        "vocab_size": int(data["vocab_size"]),
        "corpus_chars": int(data["n_chars"]),
        "d_eff": float(deff["D_eff"]),
        "d_eff_over_L": float(deff["D_eff_over_L"]),
        "F_L": float(deff["F_L"]),
        "gap_to_F_L": float(deff["gap"]),
        "rho_lag1": float(deff["rho_lag1"]),
        "rho": {str(k): float(v) for k, v in deff["rho"].items()},  # type: ignore[union-attr]
        "cka_matrix": deff["cka_matrix"],
        "norm_ratio_update_over_state": float(deff["norm_ratio_update_over_state"]),
        "n_passages_used": int(deff["n_passages_used"]),
        "n_valid_positions_per_passage": int(deff["n_valid_positions_per_passage"]),
        "val_loss": val_loss,
        "train_iters_done": len(losses),
        "train_seconds": train_seconds,
        "total_seconds": t_end - t_start,
        "measure_seconds": t_end - t_measure,
        "device": DEV,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "tf32": bool(TF32),
        "time_budget_kind": "fixed_iterations",
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    # The Runpod MCP surface has no exec channel, so stdout is the only transport
    # for results. These markers are what the operator parses; the same content
    # also exists in metrics.json on the pod volume.
    print(f"AXV_RUN_BEGIN {RUN_ID}")
    print("AXV_METRICS_JSON " + json.dumps(metrics, sort_keys=True, separators=(",", ":")))
    print(f"AXV_RUN_END {RUN_ID} status=ok")


if __name__ == "__main__":
    main()
