import enum


class AttentionBackendEnum(enum.Enum):
    FA2 = enum.auto()
    FA = enum.auto()
    SLIDING_TILE_ATTN = enum.auto()
    TORCH_SDPA = enum.auto()
    SAGE_ATTN = enum.auto()
    SAGE_ATTN_3 = enum.auto()
    VIDEO_SPARSE_ATTN = enum.auto()
    VMOBA_ATTN = enum.auto()
    AITER = enum.auto()
    NO_ATTENTION = enum.auto()

    def __str__(self):
        return self.name.lower()
