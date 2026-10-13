"""Decode context parallel primitives (layout + LSE merge).

Keep this package free of ``server_args`` so importers of the math do not
pull the CLI graph (same split CUDA uses for ``layers.dcp``).
"""

from sgl_jax.srt.layers.dcp.comm import gather_merge_dcp_attention
from sgl_jax.srt.layers.dcp.layout import (
    attention_tp_size,
    get_dcp_lens,
    owner,
    physical_index,
    physical_page_indices,
    physical_write_loc,
    validate_dcp_mesh,
    virtual_index,
    virtual_page_size,
)
from sgl_jax.srt.layers.dcp.merge import merge_dcp_attention, merge_dcp_attention_jax

__all__ = [
    "attention_tp_size",
    "gather_merge_dcp_attention",
    "get_dcp_lens",
    "merge_dcp_attention",
    "merge_dcp_attention_jax",
    "owner",
    "physical_index",
    "physical_page_indices",
    "physical_write_loc",
    "validate_dcp_mesh",
    "virtual_index",
    "virtual_page_size",
]
