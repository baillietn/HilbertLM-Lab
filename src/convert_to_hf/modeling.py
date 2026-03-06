import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class HilbertLMConfig(PretrainedConfig):
    model_type = "HilbertLM"

    def __init__(
        self,
        vocab_size=49152,
        hidden_size=576,           
        num_hidden_layers=30,      
        num_attention_heads=9,     
        num_key_value_heads=3,     
        block_size=2048,
        use_layernorm=False,
        use_swiglu=True,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.block_size = block_size
        self.use_layernorm = use_layernorm
        self.use_swiglu = use_swiglu

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

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
        seq_len = x.shape[2]
        cos = self.cos[:, :, :seq_len, :].to(x.dtype)
        sin = self.sin[:, :, :seq_len, :].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated_half = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (x_rotated_half * sin)
    
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(x) * gate

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, max_len, num_key_value_heads, use_layernorm=False, use_swiglu=True):
        super().__init__()
        self.n_head = num_attention_heads
        self.n_kv_head = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.hidden_size = hidden_size

        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        total_qkv_dim = self.q_size + 2 * self.kv_size

        self.rope = RoPE(self.head_dim, max_len)
        ffn_hidden = int(hidden_size * 8/3) if use_swiglu else int(hidden_size * 4)

        self.ln1 = nn.LayerNorm(hidden_size) if use_layernorm else nn.RMSNorm(hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, total_qkv_dim, bias=False)
        self.c_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.ln2 = nn.LayerNorm(hidden_size) if use_layernorm else nn.RMSNorm(hidden_size)

        if use_swiglu:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, 2 * ffn_hidden, bias=False),
                SwiGLU(), 
                nn.Linear(ffn_hidden, hidden_size, bias=False)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, ffn_hidden, bias=False),
                nn.GELU(),
                nn.Linear(ffn_hidden, hidden_size, bias=False)
            )
                
    def forward(self, x):
        residual = x
        
        x_norm = self.ln1(x)
        qkv = self.qkv_proj(x_norm)
        
        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=2)
        B, T, _ = q.size()
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        
        x = residual + self.c_proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        
        return x

class HilbertLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads, max_len, num_key_value_heads, use_layernorm=False, use_swiglu=True):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_attention_heads, max_len, num_key_value_heads, use_layernorm, use_swiglu) 
            for _ in range(num_hidden_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size) if use_layernorm else nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

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

    def forward(self, x):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

class HilbertLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = HilbertLMConfig
    _keys_to_ignore_on_load_missing = ["model.lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config 
        
        self.model = HilbertLM(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            max_len=config.block_size,
            num_key_value_heads=config.num_key_value_heads,
            use_layernorm=config.use_layernorm,
            use_swiglu=config.use_swiglu
        )
        
        if config.tie_word_embeddings:
            self.all_tied_weights_keys = {"model.token_embedding.weight": "model.lm_head.weight"}
        else:
            self.all_tied_weights_keys = {}
    
    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        if self.config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.token_embedding.weight
    
    def get_input_embeddings(self):
        return self.model.token_embedding
    
    def set_input_embeddings(self, value):
        self.model.token_embedding = value
    
    def get_output_embeddings(self):
        return self.model.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}