"""PD (Prefill-Decode) disaggregation infrastructure.

Stage 0 introduces the transfer wrapper, the connection ABC, and a single
backend (jax_transfer). Subsequent stages add host pinned pool / side
channel (Stage 1), bootstrap + scheduler integration (Stage 2), and
multi-host routing (Stage 3).
"""
