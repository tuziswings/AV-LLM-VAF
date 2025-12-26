import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Ensure AVCD modules are importable when running from ClearSight
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
VIDEOLLAMA_ROOT = WORKSPACE_ROOT / "AVCD"
if str(VIDEOLLAMA_ROOT) not in sys.path:
    sys.path.append(str(VIDEOLLAMA_ROOT))

from videollama2.model.qwen import (
    Qwen2Attention,
    apply_rotary_pos_emb as qwen_apply_rotary_pos_emb,
    repeat_kv as qwen_repeat_kv,
)


@dataclass
class SegmentInfo:
    system: int
    visual: int
    audio: int
    text: int


class AttnAdapter(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn_map = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: Optional[int] = None,
        modality: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    "AttnAdapter requires layer_idx when using cache - pass layer_idx"
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = qwen_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = qwen_repeat_kv(key_states, self.num_key_value_groups)
        value_states = qwen_repeat_kv(value_states, self.num_key_value_groups)

        scale_factor = 1.0 / math.sqrt(self.head_dim)
        
        chunk_size = 512
        q_len = query_states.shape[2]
        kv_len = key_states.shape[2]
        
        if q_len * kv_len < 1000000:
            with torch.cuda.amp.autocast(enabled=False):
                query_states_fp32 = query_states.to(torch.float32)
                key_states_fp32 = key_states.to(torch.float32)
                attn_weights = torch.matmul(query_states_fp32, key_states_fp32.transpose(2, 3)) * scale_factor
                del query_states_fp32, key_states_fp32
        else:
            attn_weights_chunks = []
            with torch.cuda.amp.autocast(enabled=False):
                key_states_fp32 = key_states.to(torch.float32)
                for i in range(0, q_len, chunk_size):
                    chunk_end = min(i + chunk_size, q_len)
                    query_chunk_fp32 = query_states[:, :, i:chunk_end, :].to(torch.float32)
                    chunk_weights = torch.matmul(query_chunk_fp32, key_states_fp32.transpose(2, 3)) * scale_factor
                    attn_weights_chunks.append(chunk_weights)
                    del query_chunk_fp32
                    torch.cuda.empty_cache()
                attn_weights = torch.cat(attn_weights_chunks, dim=2)
                del key_states_fp32, attn_weights_chunks

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            del causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        if self.attn_map is None or self.attn_map.shape != attn_weights.shape:
            self.attn_map = torch.ones_like(attn_weights, requires_grad=True)
        attn_weights = attn_weights * self.attn_map

        with torch.cuda.amp.autocast(enabled=False):
            value_states_fp32 = value_states.to(torch.float32)
            attn_weights_fp32 = attn_weights.to(torch.float32)
            attn_output = torch.matmul(attn_weights_fp32, value_states_fp32)
            attn_output = attn_output.to(query_states.dtype)
            del value_states_fp32, attn_weights_fp32

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not use_cache:
            past_key_value = None

        if modality is None and attn_output.size(1) != 1:
            last_query = attn_weights[:, :, -1, :].unsqueeze(-1).to(torch.float32)
            return attn_output, attn_weights if output_attentions else None, past_key_value, [last_query]

        return_attn_weights = attn_weights if output_attentions else None
        return attn_output, return_attn_weights, past_key_value


def _segment_offsets(segment_info: SegmentInfo) -> tuple[int, int, int, int]:
    sys_end = segment_info.system
    vis_end = sys_end + segment_info.visual
    aud_end = vis_end + segment_info.audio
    txt_end = aud_end + segment_info.text
    return sys_end, vis_end, aud_end, txt_end


def saliency_compute(saliency: torch.Tensor, segment_info: SegmentInfo) -> list[float]:
    saliency = torch.mean(saliency, dim=1)
    assert len(saliency.shape) == 3 and saliency.shape[0] == 1
    saliency = saliency.squeeze(0).numpy()

    sys_end, vis_end, aud_end, txt_end = _segment_offsets(segment_info)

    vis_len = max(segment_info.visual, 1)
    aud_len = max(segment_info.audio, 1)
    txt_len = max(segment_info.text, 1)

    vis_slice = slice(sys_end, vis_end)
    aud_slice = slice(vis_end, aud_end)
    txt_slice = slice(aud_end, txt_end)

    def safe_avg(block, denom):
        if block.size == 0 or denom == 0:
            return 0.0
        return float(block.sum() / denom)

    vis_vis = safe_avg(
        saliency[vis_slice, vis_slice],
        ((vis_len + 1) * vis_len / 2) if vis_len > 1 else vis_len,
    )
    
    txt_vis = safe_avg(saliency[txt_slice, vis_slice], vis_len * txt_len)
    
    vis_txt = safe_avg(saliency[vis_slice, txt_slice], vis_len * txt_len)
    
    aud_vis = safe_avg(saliency[aud_slice, vis_slice], vis_len * aud_len)
    
    vis_aud = safe_avg(saliency[vis_slice, aud_slice], vis_len * aud_len)
    
    txt_aud = safe_avg(saliency[txt_slice, aud_slice], aud_len * txt_len)
    
    aud_txt = safe_avg(saliency[aud_slice, txt_slice], aud_len * txt_len)

    return [vis_vis, txt_vis, vis_txt, aud_vis, vis_aud, txt_aud, aud_txt]


def attention_compute(attention: torch.Tensor, segment_info: SegmentInfo) -> list[float]:
    attention = torch.mean(attention, dim=1)
    assert len(attention.shape) == 3 and attention.shape[0] == 1
    attention = attention.squeeze(0).numpy()

    sys_end, vis_end, aud_end, txt_end = _segment_offsets(segment_info)

    props_sys = attention[-1][:segment_info.system].sum()
    props_vis = attention[-1][sys_end:vis_end].sum()
    props_aud = attention[-1][vis_end:aud_end].sum()
    props_txt = attention[-1][aud_end:txt_end].sum()

    return [float(props_sys), float(props_vis), float(props_aud), float(props_txt)]
