"""
Tokenisation & dataset loader for the RPlan floor-plan corpus.

It re-uses the (x,y) → point-token mapping logic already defined in
`TokenizationSchema` (see swiss_data_coordtok.py), but swaps in the eight
RPlan room categories and a far simpler polygon-to-token pipeline: we assume
the JSON room polygons are already axis-aligned rectangles, so we do **not**
run the heavy rectilinear checks that depended on Shapely, etc.
"""

from __future__ import annotations
import json
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 1.  ROOM-TYPE DICTIONARY  (RPlan specific)
# ---------------------------------------------------------------------------
room_names: Dict[int, str] = {
    1: "Living room",
    2: "Kitchen",
    3: "Bedroom",
    4: "Bathroom",
    5: "Balcony",
    6: "Dining room",
    7: "Study room",
    8: "Storage",
}

# Build <name> ↔ index maps (0-based indices for token offsetting)
NameToIndex = {v.upper().replace(" ", "_"): i for i, v in room_names.items()}
IndexToName = {i: n for n, i in NameToIndex.items()}

# ---------------------------------------------------------------------------
# 2.  TOKENISATION SCHEMA (self-contained – no swiss_data_coordtok dependency)
# ---------------------------------------------------------------------------

class TokenizationSchema:
    """Minimal standalone schema for point-token representation used by RPlan."""

    def __init__(self, discretization_factor: float = 0.5, max_size: float = 256.0):
        self.DISCRETIZATION_FACTOR = discretization_factor
        self.MAX_COORD_LENGTH_VALUE = max_size - discretization_factor
        self.COORD_LENGTH_RANGE = int(self.MAX_COORD_LENGTH_VALUE / discretization_factor) + 1
        self.COORD_LENGTH_MAX_VALUE = self.COORD_LENGTH_RANGE - 1

        # Point tokens occupy the first continuous range of the vocabulary.
        self.NUM_POINT_TOKENS = self.COORD_LENGTH_RANGE * self.COORD_LENGTH_RANGE

        # Special tokens
        self.BOS_TOKEN = self.NUM_POINT_TOKENS
        self.EOS_TOKEN = self.NUM_POINT_TOKENS + 1
        self.PAD_TOKEN = self.NUM_POINT_TOKENS + 2

        # Room-name (feature) tokens start here
        self.NAME_START_INDEX = self.NUM_POINT_TOKENS + 3

        # Final vocab size (point tokens + special + names)
        self.VOCAB_SIZE = self.NAME_START_INDEX + len(NameToIndex)

        # Sequence length upper bound (can be overwritten by caller)
        self.MAX_SEQ_LEN = 512

    # ------------------------------------------------------------------ #
    #  Point token helpers
    # ------------------------------------------------------------------ #
    def point_to_token(self, x_index: int, y_index: int) -> int:
        if not (0 <= x_index <= self.COORD_LENGTH_MAX_VALUE) or not (
            0 <= y_index <= self.COORD_LENGTH_MAX_VALUE
        ):
            raise ValueError(
                f"Discrete coordinates ({x_index},{y_index}) out of range [0,{self.COORD_LENGTH_MAX_VALUE}]"
            )
        return y_index * self.COORD_LENGTH_RANGE + x_index

    def token_to_point(self, token: int) -> tuple[int, int]:
        if not (0 <= token < self.NUM_POINT_TOKENS):
            raise ValueError(
                f"Point token {token} out of range [0,{self.NUM_POINT_TOKENS - 1}]"
            )
        y_index = token // self.COORD_LENGTH_RANGE
        x_index = token % self.COORD_LENGTH_RANGE
        return x_index, y_index

# Replace external dependency with the local schema
DEFAULT_SCHEMA = TokenizationSchema()

# ---------------------------------------------------------------------------
# 3.  SMALL HELPERS
# ---------------------------------------------------------------------------
def discretise(coord: float, schema: TokenizationSchema) -> int:
    """Float → discrete grid index, clamped to the valid range."""
    disc = int(round(coord / schema.DISCRETIZATION_FACTOR))
    return max(0, min(disc, schema.COORD_LENGTH_MAX_VALUE))


def polygon_to_tokens(
    polygon: List[List[float]], schema: TokenizationSchema
) -> List[int]:
    """
    Convert an RPlan polygon (list[[x,y], ...]) into point tokens.

    If the last vertex duplicates the first (a common convention for closed
    polygons) it is dropped to avoid an unnecessary token.
    """
    if len(polygon) > 2 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]  # drop duplicate closing vertex

    tokens = [
        schema.point_to_token(discretise(x, schema), discretise(y, schema))
        for x, y in polygon
    ]
    return tokens


# ---------------------------------------------------------------------------
# 4.  DATASET
# ---------------------------------------------------------------------------
class RPlanDataset(Dataset):
    """
    Yields dicts with 'input_ids', 'labels', 'attention_mask' suitable for
    masked-language modelling / diffusion SFT.

    Sequence template:
      [BOS]  pt_tok...  <NAME_tok>   ...   [EOS]   [PAD]...
    """

    def __init__(
        self,
        root_dir: str,
        tokenizer: TokenizationSchema | None = None,
        max_seq_len: int = 512,
        augment: bool = False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.files = sorted(
            f
            for f in os.listdir(root_dir)
            if f.endswith(".json") and f.split(".")[0].isdigit()
        )
        if not self.files:
            raise RuntimeError(f"No *.json RPlan files found in {root_dir}")

        self.schema = tokenizer or DEFAULT_SCHEMA
        # Allow per-dataset override
        self.schema.MAX_SEQ_LEN = max_seq_len

    @staticmethod
    def _augment_polygons(polygons: list[list[list[float]]]) -> list[list[list[float]]]:
        """Applies random rotation and flipping to a set of polygons."""
        if not polygons:
            return polygons
        
        # Convert to numpy for easier manipulation
        np_polygons = [np.array(p) for p in polygons]
        
        # --- Center the whole floorplan before transformations ---
        all_points = np.vstack(np_polygons)
        min_coords = all_points.min(axis=0)
        
        # Reposition to origin
        np_polygons = [(p - min_coords) for p in np_polygons]

        # --- 1. Random Rotation (90, 180, 270 degrees) ---
        k = random.randint(0, 3)
        if k > 0:
            # Rotation matrices for multiples of 90 degrees are simple
            rot_matrices = [
                np.array([[0, -1], [1, 0]]),   # 90 deg
                np.array([[-1, 0], [0, -1]]),  # 180 deg
                np.array([[0, 1], [-1, 0]])    # 270 deg
            ]
            # matrix multiplication for rotation
            np_polygons = [p @ rot_matrices[k-1] for p in np_polygons]

        # --- 2. Random Flip (Horizontally) ---
        if random.random() < 0.5:
            # Flip x-coordinates
            for p in np_polygons:
                p[:, 0] = -p[:, 0]

        # --- Reposition back to positive coordinates ---
        all_points = np.vstack(np_polygons)
        min_coords = all_points.min(axis=0)
        np_polygons = [(p - min_coords) for p in np_polygons]

        return [p.tolist() for p in np_polygons]

    # ------------------------------------------------------------------ #
    #  Core encoding routine
    # ------------------------------------------------------------------ #
    def _generate_caption(self, room_types: List[int]) -> str:
        """Generate a simple natural-language caption from the list of room type IDs."""
        type_counts: Dict[int, int] = {}
        for t in room_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        parts: List[str] = []
        total = len(room_types)
        parts.append(f"The floorplan contains {total} rooms.")
        for t_id, count in sorted(type_counts.items()):
            name = room_names.get(t_id, "Unknown room")
            noun = name.lower()
            plural = "s" if count > 1 else ""
            parts.append(f"{count} {noun}{plural}")
        return " ".join(parts)

    def _encode_sample(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        all_tokens: List[int] = [self.schema.BOS_TOKEN]

        polygons: List[List[List[float]]] = data["room_polygons"]
        room_types: List[int] = data["rms_type"]

        if self.augment:
            polygons = self._augment_polygons(polygons)

        for poly, r_type in zip(polygons, room_types):
            #   1) polygon vertices → point tokens
            all_tokens.extend(polygon_to_tokens(poly, self.schema))

            #   2) append NAME token (offset onto schema NAME space)
            room_name = room_names.get(r_type, "UNKNOWN").upper().replace(" ", "_")
            name_idx = NameToIndex.get(room_name, -1)
            if name_idx == -1:
                raise ValueError(f"Room type {r_type} not in room_names dict.")
            all_tokens.append(self.schema.NAME_START_INDEX + name_idx)

        # Close sequence
        all_tokens.append(self.schema.EOS_TOKEN)

        # Pad / truncate
        if len(all_tokens) > self.schema.MAX_SEQ_LEN:
            all_tokens = all_tokens[: self.schema.MAX_SEQ_LEN]
            all_tokens[-1] = self.schema.EOS_TOKEN
        pad_len = self.schema.MAX_SEQ_LEN - len(all_tokens)
        all_tokens.extend([self.schema.PAD_TOKEN] * pad_len)

        input_ids = torch.tensor(all_tokens, dtype=torch.long)
        attention_mask = (input_ids != self.schema.PAD_TOKEN).long()
        labels = input_ids.clone()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            caption=self._generate_caption(room_types),
        )

    # ------------------------------------------------------------------ #
    #  PyTorch Dataset hooks
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fpath = os.path.join(self.root_dir, self.files[idx])
        with open(fpath, "r") as fp:
            data = json.load(fp)

        return self._encode_sample(data)

# ---------------------------------------------------------------------------
# 5.  DECODING UTILITIES (tokens → polygons + room types)
# ---------------------------------------------------------------------------

def decode_tokens_to_polygons(
    tokens: list[int],
    schema: "TokenizationSchema",
) -> tuple[list[list[list[float]]], list[int]]:
    """Inverse of the encoder: convert a full token sequence back to polygons + class IDs.

    Returns
    -------
    polygons : list[list[[x,y], ...]]  # float coordinates in same unit as original
    room_ids : list[int]               # rms_type numbers (1..8)
    """
    polygons: list[list[list[float]]] = []
    room_ids: list[int] = []

    current_poly: list[list[float]] = []
    for t in tokens:
        if t in (schema.BOS_TOKEN, schema.PAD_TOKEN):
            continue
        if t == schema.EOS_TOKEN:
            break

        if t >= schema.NAME_START_INDEX:
            # flush current poly if any
            if current_poly:
                polygons.append(current_poly)
                current_poly = []
            # decode room class id
            cls_index = t - schema.NAME_START_INDEX
            # map back to rms_type integer (1..8)
            # IndexToName gives name string; we need reverse lookup to room id
            name_str = IndexToName[cls_index]
            # build reverse mapping once
            rms_lookup = {v.upper().replace(" ", "_"): k for k, v in room_names.items()}
            room_ids.append(rms_lookup[name_str])
        else:
            # point token
            x_idx, y_idx = schema.token_to_point(t)
            x = x_idx * schema.DISCRETIZATION_FACTOR
            y = y_idx * schema.DISCRETIZATION_FACTOR
            current_poly.append([x, y])

    # in case last poly collected but no following NAME (invalid), drop.

    return polygons, room_ids


def pretty_token(tok: int, schema: "TokenizationSchema") -> str:
    """Return a human-readable representation of one token."""
    if tok == schema.BOS_TOKEN:
        return "<BOS>"
    if tok == schema.EOS_TOKEN:
        return "<EOS>"
    if tok == schema.PAD_TOKEN:
        return "<PAD>"
    if tok >= schema.NAME_START_INDEX:
        cls_idx = tok - schema.NAME_START_INDEX
        return f"<NAME:{IndexToName[cls_idx]}>"
    x, y = schema.token_to_point(tok)
    return f"<{x},{y}>"

# Make utilities importable elsewhere
__all__ = [
    "TokenizationSchema",
    "RPlanDataset",
    "decode_tokens_to_polygons",
    "pretty_token",
    "NameToIndex",
    "IndexToName",
]
