import torch
import torch.nn as nn 
import torch.nn.functional as F

from config import config

use_layernorm = config['use_layernorm']

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, n_kv_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, n_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, n_kv_heads * n_rep, head_dim)

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        pos = torch.arange(max_seq_len, dtype=torch.float)
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(pos, theta)
        embedding = torch.cat((angles, angles), dim=-1)
        self.register_buffer('cos', embedding.cos()[None, None, :, :])
        self.register_buffer('sin', embedding.sin()[None, None, :, :])

    def forward(self, x):
        # x shape: (Batch, n_head, SeqLen, HeadDim)
        seq_len = x.shape[2]
        cos = self.cos[:, :, :seq_len, :]
        sin = self.sin[:, :, :seq_len, :]
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated_half = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated_half * sin)
    
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(x) * gate

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, max_len, n_kv_head=None, use_te=False):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.use_te = use_te

        self.n_rep = self.n_head // self.n_kv_head
        
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        total_qkv_dim = self.q_size + 2 * self.kv_size

        if use_te:
            import transformer_engine.pytorch as te
            self.ln_attn = te.LayerNormLinear(
                d_model, 
                total_qkv_dim, 
                bias=False,
                normalization="LayerNorm" if use_layernorm else "RMSNorm",
            ) 
            self.c_proj = te.Linear(d_model, d_model, bias=False)
            
            ffn_hidden = int(d_model * 8/3)
            self.ln_mlp = te.LayerNormMLP(
                hidden_size=d_model, 
                ffn_hidden_size=ffn_hidden, 
                bias=False, 
                normalization="LayerNorm" if use_layernorm else "RMSNorm",
                activation='swiglu'
            )
        else:
            self.ln1 = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)
            self.qkv_proj = nn.Linear(d_model, total_qkv_dim, bias=False)
            self.c_proj = nn.Linear(d_model, d_model, bias=False)
            
            ffn_hidden = int(d_model * 8/3)
            self.ln2 = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 2 * ffn_hidden, bias=False),
                SwiGLU(),
                nn.Linear(ffn_hidden, d_model, bias=False)
            )

        self.rope = RoPE(self.head_dim, max_len)

    def forward(self, x):
        residual = x
        
        if self.use_te:
            qkv = self.ln_attn(x)
        else:
            x_norm = self.ln1(x)
            qkv = self.qkv_proj(x_norm)
        
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=2)
        B, T, _ = q.size()
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        x = residual + self.c_proj(attn_out)

        if self.use_te:
            x = x.contiguous()
            x = x + self.ln_mlp(x)
        else:
            x = x + self.mlp(self.ln2(x))
        
        return x

class NanoLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, max_len, use_te=False, n_kv_head=None):
        super().__init__()
        self.max_len = max_len
        self.use_te = use_te
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_len, use_te=use_te, n_kv_head=n_kv_head) 
            for _ in range(n_layer)
        ])
        
        if use_te:
            import transformer_engine.pytorch as te
            self.final_norm = te.LayerNorm(d_model) if use_layernorm else te.RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            
        else:
            self.final_norm = nn.LayerNorm(d_model) if use_layernorm else nn.RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
            self.lm_head.weight = self.token_embedding.weight

            self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.token_embedding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
            return logits, loss
        
        return logits
    

