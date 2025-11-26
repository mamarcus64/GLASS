import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _tokens_from_cfg(cfg):
    P = cfg["model"]["patch_size"]
    Tp = cfg["data"]["past_len"]   # frames
    Tf = cfg["data"]["future_len"] # frames
    assert Tp % P == 0 and Tf % P == 0, "past_len/future_len must be divisible by patch_size"
    return Tp // P, Tf // P  # (Te_tokens, Td_tokens)

# ---- Positional encodings ----

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return x + self.pe(pos)

def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """RoPE for 1D sequences. Applies to q,k of shape (B, H, T, Hd)."""
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def _build_freqs(self, T: int, device):
        half = self.dim // 2
        freq_seq = torch.arange(half, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (freq_seq / half))
        t = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,f->t f", t, inv_freq)  # (T, half)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # repeat for both halves
        cos = torch.stack([cos, cos], dim=-1).reshape(T, self.dim)
        sin = torch.stack([sin, sin], dim=-1).reshape(T, self.dim)
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T, Hd), Hd == self.dim
        # cos/sin: (T, Hd) -> broadcast to (B,H,T,Hd)
        return (x * cos) + (_rotate_half(x) * sin)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B,H,T,Hd)
        B, H, T, Hd = q.shape
        assert Hd == self.dim, "RoPE dim mismatch"
        cos, sin = self._build_freqs(T, q.device)
        cos = cos[None, None, :, :]  # (1,1,T,Hd)
        sin = sin[None, None, :, :]
        return self.apply_rotary(q, cos, sin), self.apply_rotary(k, cos, sin)

# ---- Building blocks ----

class PatchEmbed(nn.Module):
    def __init__(self, D_in: int, patch_size: int, d_model: int):
        super().__init__()
        self.P = patch_size
        self.D_in = D_in
        self.proj = nn.Linear(patch_size * D_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_frames, D_in)
        B, T, D = x.shape
        assert D == self.D_in and T % self.P == 0, "Bad input shape or patch_size"
        Ttok = T // self.P
        x = x.view(B, Ttok, self.P * self.D_in)
        return self.proj(x)  # (B, Ttok, d_model)

class PatchRecover(nn.Module):
    def __init__(self, D_out: int, patch_size: int, d_model: int):
        super().__init__()
        self.P = patch_size
        self.D_out = D_out
        self.head = nn.Linear(d_model, patch_size * D_out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, Ttok, d_model)
        B, Ttok, _ = z.shape
        y = self.head(z)  # (B, Ttok, P*D_out)
        return y.view(B, Ttok * self.P, self.D_out)  # (B, T_frames, D_out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, attn_dropout, norm="preln", use_rope=False, rope_dim=None):
        super().__init__()
        self.norm_type = norm
        self.use_rope = use_rope
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.Dropout(dropout),
        )
        if use_rope:
            # Split heads for rope
            Hd = d_model // n_heads
            rope_dim = Hd if rope_dim is None else rope_dim
            rope_dim = rope_dim - (rope_dim % 2)
            self.rope_dim = rope_dim
            self.rope = RotaryEmbedding(rope_dim)
        else:
            self.rope = None

    def _apply_rope_to_qk(self, q, k, n_heads):
        # q,k: (B, T, d_model) -> (B,H,T,Hd)
        B, T, D = q.shape
        Hd = D // n_heads
        q = q.view(B, T, n_heads, Hd).transpose(1, 2)  # (B,H,T,Hd)
        k = k.view(B, T, n_heads, Hd).transpose(1, 2)
        # Apply RoPE only to first rope_dim dims of Hd
        if self.rope_dim < Hd:
            q_ro, q_tail = q[..., :self.rope_dim], q[..., self.rope_dim:]
            k_ro, k_tail = k[..., :self.rope_dim], k[..., self.rope_dim:]
            q_ro, k_ro = self.rope(q_ro, k_ro)
            q = torch.cat([q_ro, q_tail], dim=-1)
            k = torch.cat([k_ro, k_tail], dim=-1)
        else:
            q, k = self.rope(q, k)
        # back to (B,T,D)
        q = q.transpose(1, 2).contiguous().view(B, T, D)
        k = k.transpose(1, 2).contiguous().view(B, T, D)
        return q, k

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, kv: Optional[torch.Tensor] = None):
        # x: (B,T,D). If kv is provided, use as K/V (for cross-attention in decoder).
        residual = x
        if self.norm_type == "preln":
            x = self.ln1(x)

        q = x
        k = x if kv is None else kv
        v = x if kv is None else kv

        # Apply RoPE on q,k if enabled
        if self.use_rope:
            # MultiheadAttention doesn't expose q,k pre-proj easily, so we emulate timing by
            # applying rope in model space; this is an approximation but works well in practice.
            q, k = self._apply_rope_to_qk(q, k, self.attn.num_heads)

        x, _ = self.attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = F.dropout(x, p=self.mlp[-1].p, training=self.training)  # reuse dropout value
        x = residual + x

        residual = x
        if self.norm_type == "preln":
            x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        if self.norm_type == "postln":
            x = self.ln2(self.ln1(x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, attn_dropout, layers, norm, pos_enc, max_tokens: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, attn_dropout, norm=norm, use_rope=(pos_enc=="rope"))
            for _ in range(layers)
        ])
        self.final_ln = nn.LayerNorm(d_model) if norm == "preln" else nn.Identity()
        self.pos_enc_type = pos_enc
        self.learned_pe = LearnedPositionalEncoding(max_tokens, d_model) if pos_enc == "learned" else None

    def forward(self, x: torch.Tensor):
        if self.learned_pe is not None:
            # learned_pe supports any T up to max_tokens
            x = self.learned_pe(x)
        for blk in self.layers:
            x = blk(x)
        x = self.final_ln(x)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, attn_dropout, layers, norm, pos_enc, max_tokens: int):
        super().__init__()
        self.self_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, attn_dropout, norm=norm, use_rope=(pos_enc=="rope"))
            for _ in range(layers)
        ])
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, attn_dropout, norm=norm, use_rope=False)
            for _ in range(layers)
        ])
        self.final_ln = nn.LayerNorm(d_model) if norm == "preln" else nn.Identity()
        self.pos_enc_type = pos_enc
        self.learned_pe = LearnedPositionalEncoding(max_tokens, d_model) if pos_enc == "learned" else None

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor, tgt_mask: torch.Tensor, tgt_key_padding_mask: Optional[torch.Tensor] = None):
        if self.learned_pe is not None:
            x = self.learned_pe(x)
        for self_blk, cross_blk in zip(self.self_blocks, self.cross_blocks):
            x = self_blk(x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            x = cross_blk(x, kv=enc_out)
        x = self.final_ln(x)
        return x

class PatchSeq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ---- read cfg ----
        P          = int(cfg["model"]["patch_size"])
        D_in       = int(cfg["model"]["D_in"])
        d_model    = int(cfg["model"]["d_model"])
        n_heads    = int(cfg["model"]["n_heads"])
        enc_layers = int(cfg["model"]["enc_layers"])
        dec_layers = int(cfg["model"]["dec_layers"])
        mlp_ratio  = float(cfg["model"]["mlp_ratio"])
        dropout    = float(cfg["model"]["dropout"])
        attn_drop  = float(cfg["model"]["attn_dropout"])
        norm       = cfg["model"].get("norm", "preln")
        pos_enc    = cfg["model"].get("pos_enc", "rope")  # "rope" or "learned"
        share_pe   = bool(cfg["model"].get("share_patch_embed", False))
        self.out_clamp = bool(cfg.get("loss", {}).get("clamp_outputs", False))

        Te_tokens, Td_tokens = _tokens_from_cfg(cfg)
        self.P = P
        self.D = D_in
        self.Td_tokens = Td_tokens

        # ---- modules ----
        self.patch_embed_enc = PatchEmbed(D_in, P, d_model)
        self.patch_embed_dec = self.patch_embed_enc if share_pe else PatchEmbed(D_in, P, d_model)

        self.encoder = Encoder(
            d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
            dropout=dropout, attn_dropout=attn_drop, layers=enc_layers,
            norm=norm, pos_enc=pos_enc, max_tokens=Te_tokens
        )
        self.decoder = Decoder(
            d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
            dropout=dropout, attn_dropout=attn_drop, layers=dec_layers,
            norm=norm, pos_enc=pos_enc, max_tokens=Td_tokens
        )

        self.recover = PatchRecover(D_out=D_in, patch_size=P, d_model=d_model)

        # scheduled sampling helper
        self.stop_grad_pred_embed = bool(
            cfg.get("optim", {}).get("sched_sampling", {}).get("stop_grad_pred_embed", True)
        )

    # --- utils ---
    def _make_causal_mask(self, T: int, device):
        m = torch.full((T, T), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    # --- scheduled sampling / AR token loop ---
    def _decode_token_loop(self, enc_out: torch.Tensor, future_frames: torch.Tensor, tf_prob: float) -> torch.Tensor:
        """
        tf_prob in [0,1]; 1.0 = always ground-truth (TF), 0.0 = always predicted (pure AR).
        """
        B = enc_out.size(0)
        device = enc_out.device

        # GT tokens for optional teacher forcing steps
        dec_tokens_truth = self.patch_embed_dec(future_frames)  # (B, Td, d_model)

        # BOS = zero token
        dec_seq = torch.zeros(B, 1, enc_out.size(-1), device=device)
        preds_blocks = []

        for t in range(self.Td_tokens):
            tgt_mask = self._make_causal_mask(dec_seq.size(1), device=device)
            dec_out = self.decoder(dec_seq, enc_out, tgt_mask=tgt_mask)      # (B, t+1, d_model)
            last = dec_out[:, -1:, :]                                        # (B,1,d_model)
            frames_block = self.recover(last)                                 # (B, P, D)
            preds_blocks.append(frames_block)

            # choose next token (GT vs predicted)
            frames_for_embed = frames_block.detach() if self.stop_grad_pred_embed else frames_block
            pred_tok = self.patch_embed_dec(frames_for_embed)                 # (B,1,d_model)
            gt_tok  = dec_tokens_truth[:, t:t+1, :]                           # (B,1,d_model)

            if tf_prob <= 0.0:
                next_tok = pred_tok
            elif tf_prob >= 1.0:
                next_tok = gt_tok
            else:
                keep_gt = (torch.rand(B, 1, 1, device=device) < tf_prob).float()
                next_tok = keep_gt * gt_tok + (1.0 - keep_gt) * pred_tok

            dec_seq = torch.cat([dec_seq, next_tok], dim=1)

        pred_future = torch.cat(preds_blocks, dim=1)                           # (B, Tf, D)
        if self.out_clamp:
            pred_future = torch.clamp(pred_future, -10.0, 10.0)
        return pred_future

    # --- forward ---
    def forward(self, past_frames: torch.Tensor, future_frames: torch.Tensor | None = None, tf_prob: float | None = None) -> torch.Tensor:
        """
        If tf_prob is None or >=1.0: classic teacher-forced parallel decode.
        Else: scheduled sampling / AR loop with tf_prob in [0,1].
        """
        enc_tokens = self.patch_embed_enc(past_frames)
        enc_out = self.encoder(enc_tokens)

        if tf_prob is None or tf_prob >= 1.0:
            assert future_frames is not None, "future_frames required for teacher-forced forward"
            dec_tokens_in = self.patch_embed_dec(future_frames)
            zero_tok = torch.zeros(future_frames.size(0), 1, dec_tokens_in.size(-1), device=past_frames.device)
            dec_in = torch.cat([zero_tok, dec_tokens_in[:, :-1, :]], dim=1)   # shift-right with BOS
            tgt_mask = self._make_causal_mask(dec_in.size(1), device=past_frames.device)
            dec_out = self.decoder(dec_in, enc_out, tgt_mask=tgt_mask)
            pred_future = self.recover(dec_out)
            if self.out_clamp:
                pred_future = torch.clamp(pred_future, -10.0, 10.0)
            return pred_future

        # scheduled sampling / AR
        assert future_frames is not None, "future_frames required to build GT tokens during scheduled sampling"
        return self._decode_token_loop(enc_out, future_frames, tf_prob)

    @torch.no_grad()
    def generate(self, past_frames: torch.Tensor) -> torch.Tensor:
        """Pure AR rollout (equivalent to tf_prob=0)."""
        self.eval()
        enc_tokens = self.patch_embed_enc(past_frames)
        enc_out = self.encoder(enc_tokens)
        dummy_future = torch.zeros(past_frames.size(0), self.Td_tokens * self.P, self.D, device=past_frames.device)
        return self._decode_token_loop(enc_out, dummy_future, tf_prob=0.0)
