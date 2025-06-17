import logging
import multiprocessing as mp
import os
import random
import re
import shutil
from collections import defaultdict
from pprint import pprint
from typing import Any
import json
import copy

import numpy as np
import shapely
import torch
from torch.utils.data import Dataset

from .swiss_fp_data import fp_utils as Mapping
from .swiss_fp_data.floorplan_data import FloorplanData, load_floorplan
from .tokenization_schema import BaseTokenizationSchema
from .utility_functions import RPolygon

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


FEATURES_TO_PROCESS: set[str] = {
    "boundaries",
    "spaces",
    "door_windows",
}


class PreprocessingError(Exception):
    """Custom exception for errors during data preprocessing."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


# Add new exception for sequence length
class SequenceTooLongError(PreprocessingError):
    """Custom exception for sequences exceeding max_length."""

    def __init__(self, key: str, length: int, max_length: int):
        message = (
            f"Sequence length {length} exceeds max_length {max_length} for sample {key}"
        )
        super().__init__(message)
        self.key = key
        self.length = length
        self.max_length = max_length


def make_it_rectilinear(arr: np.ndarray) -> np.ndarray:
    """Ensures a polygon is rectilinear and returns its vertices."""
    arr = arr[..., :2]

    # Use buffer(0) to clean up potential self-intersections or invalid geometries
    try:
        pol = RPolygon(arr).buffer(0)
    except Exception as buffer_err:
        raise PreprocessingError(f"Polygon could not be buffered: {buffer_err}")

    # Check the result of buffer(0)
    if not isinstance(pol, shapely.Polygon) or pol.is_empty:
        # If buffer(0) resulted in non-Polygon or empty, raise error
        raise PreprocessingError("Polygon becomes invalid or empty after buffer(0)")

    arr = np.array(pol.exterior.coords)

    # Check if the polygon has at least 3 vertices after buffer
    if len(arr) < 4:  # Need at least 3 vertices + closing point
        raise PreprocessingError(
            f"Polygon has too few vertices after buffer(0): {len(arr) - 1}"
        )

    # Check angles - Use a tolerance
    tolerance = 1e-6
    for i in range(len(arr) - 1):
        p1 = arr[i]
        p2 = arr[(i + 1)]
        p3 = arr[(i + 2) % (len(arr) - 1)]  # Use modulo on vertex count

        vec1 = p2 - p1
        vec2 = p3 - p2

        dot_product = np.dot(vec1, vec2)

        # Check for perpendicularity (dot product close to zero)
        # Check for collinearity (cross product close to zero) - skip collinear check here
        if abs(dot_product) > tolerance * np.linalg.norm(vec1) * np.linalg.norm(vec2):
            # Check if angle is multiple of 90 degrees more robustly
            angle_deg = np.degrees(
                np.arccos(
                    np.clip(
                        dot_product / (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                        -1.0,
                        1.0,
                    )
                )
            )
            if not np.isclose(angle_deg % 90, 0, atol=1e-3) and not np.isclose(
                angle_deg % 90, 90, atol=1e-3
            ):
                raise PreprocessingError(
                    f"Non-rectilinear angle ({angle_deg} deg) detected between segments {p1}-{p2} and {p2}-{p3}"
                )

    return arr[:-1]  # Return without the closing point


def reposition(points: np.ndarray) -> np.ndarray:
    """Translates points so that the minimum coordinates are at (0,0)."""
    min_coords = np.min(points, axis=0)
    return points - min_coords


def random_rotate(points: np.ndarray) -> np.ndarray:
    """Rotates points by a random angle of 90, 180, or 270 degrees."""
    rotation_matrices = {
        90: np.array([[0, -1], [1, 0]]),
        180: np.array([[-1, 0], [0, -1]]),
        270: np.array([[0, 1], [-1, 0]]),
    }
    angle = random.choice([90, 180, 270])
    return np.dot(np.asarray(points), rotation_matrices[angle])


def random_flip(points: np.ndarray) -> np.ndarray:
    """Randomly flips points horizontally and/or vertically."""
    if random.random() > 0.5:
        points[:, 0] = -points[:, 0]  # Flip horizontally
    if random.random() > 0.5:
        points[:, 1] = -points[:, 1]  # Flip vertically
    return points


class TokenizationSchema(BaseTokenizationSchema):
    def __init__(self, discretization_factor: float = 0.05, max_size: float = 25.6):
        super().__init__()
        self.max_size = max_size
        self.DISCRETIZATION_FACTOR = discretization_factor
        self.MAX_COORD_LENGTH_VALUE = max_size - discretization_factor
        self.COORD_LENGTH_RANGE = (
            int(self.MAX_COORD_LENGTH_VALUE / discretization_factor) + 1
        )
        self.COORD_LENGTH_MAX_VALUE = self.COORD_LENGTH_RANGE - 1

        # New: Point token definitions
        self.NUM_POINT_TOKENS = self.COORD_LENGTH_RANGE * self.COORD_LENGTH_RANGE
        # Point tokens will range from 0 to NUM_POINT_TOKENS - 1

        self.BOS_TOKEN = self.NUM_POINT_TOKENS
        self.EOS_TOKEN = self.NUM_POINT_TOKENS + 1
        self.PAD_TOKEN = self.NUM_POINT_TOKENS + 2
        # Removed NORTH_TOKEN, WEST_TOKEN, EAST_TOKEN, SOUTH_TOKEN

        self.NAME_START_INDEX = self.NUM_POINT_TOKENS + 3
        self.VOCAB_SIZE = self.NAME_START_INDEX + len(Mapping.IndexToName)
        self.MAX_SEQ_LEN = 512 # This might need re-evaluation

        self.NameMap = sorted(list(Mapping.IndexToName.values()))
        self.NameToIndexMap = {n: i for i, n in enumerate(self.NameMap)}
        self.IndexToNameMap = {i: n for i, n in enumerate(self.NameMap)}
        self.vocab_size = self.VOCAB_SIZE

    def point_to_token(self, x_discrete: int, y_discrete: int) -> int:
        """Converts a discretized (x,y) coordinate pair to a single point token."""
        if not (0 <= x_discrete <= self.COORD_LENGTH_MAX_VALUE and \
                0 <= y_discrete <= self.COORD_LENGTH_MAX_VALUE):
            raise ValueError(
                f"Discrete coordinates ({x_discrete}, {y_discrete}) out of range [0, {self.COORD_LENGTH_MAX_VALUE}]"
            )
        return y_discrete * self.COORD_LENGTH_RANGE + x_discrete

    def token_to_point(self, token: int) -> tuple[int, int]:
        """Converts a single point token back to a discretized (x,y) coordinate pair."""
        if not (0 <= token < self.NUM_POINT_TOKENS):
            raise ValueError(
                f"Point token {token} out of range [0, {self.NUM_POINT_TOKENS - 1}]"
            )
        y_discrete = token // self.COORD_LENGTH_RANGE
        x_discrete = token % self.COORD_LENGTH_RANGE
        return x_discrete, y_discrete

    def __str__(self):
        return (
            f"TokenizationSchema(\\n"
            f"    DISCRETIZATION_FACTOR: {self.DISCRETIZATION_FACTOR}\\n"
            f"    MAX_COORD_LENGTH_VALUE: {self.MAX_COORD_LENGTH_VALUE}\\n"
            f"    COORD_LENGTH_RANGE: {self.COORD_LENGTH_RANGE}\\n"
            f"    COORD_LENGTH_MAX_VALUE: {self.COORD_LENGTH_MAX_VALUE}\\n"
            f"    NUM_POINT_TOKENS: {self.NUM_POINT_TOKENS}\\n"
            f"    BOS_TOKEN: {self.BOS_TOKEN}\\n"
            f"    EOS_TOKEN: {self.EOS_TOKEN}\\n"
            f"    PAD_TOKEN: {self.PAD_TOKEN}\\n"
            f"    NAME_START_INDEX: {self.NAME_START_INDEX}\\n"
            f"    VOCAB_SIZE: {self.VOCAB_SIZE}\\n"
            f"    MAX_SEQ_LEN: {self.MAX_SEQ_LEN}\\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()

    def detokenize(self, tokens: torch.Tensor) -> tuple[list[int], list[np.ndarray]]:
        """
        Converts a token sequence back into feature names and polygon coordinates.

        Args:
            tokens: The 1D tensor of tokens.

        Returns:
            A tuple containing:
            - list[int]: A list of name indices (0-based relative to name mapping).
            - list[np.ndarray]: A list of corresponding polygon coordinates (Nx2).
        """
        name_start_index = self.NAME_START_INDEX
        eos_token = self.EOS_TOKEN
        pad_token = self.PAD_TOKEN
        bos_token = self.BOS_TOKEN
        discretization_factor = self.DISCRETIZATION_FACTOR
        # coord_length_max_value = self.COORD_LENGTH_MAX_VALUE # Not directly used here now

        reconstructed_polygons = []
        names = []
        if tokens.device.type != "cpu":
            tokens = tokens.cpu()
        token_list = tokens.tolist()

        seq_end_idx = len(token_list)
        for i, t in enumerate(token_list):
            if t in (eos_token, pad_token):
                seq_end_idx = i
                break
        active_tokens = token_list[:seq_end_idx]

        if active_tokens and active_tokens[0] == bos_token:
            active_tokens = active_tokens[1:]

        name_indices_in_seq = [i for i, t in enumerate(active_tokens) if t >= name_start_index]
        segment_starts = [0] + [idx + 1 for idx in name_indices_in_seq[:-1]]

        for i, name_idx_pos in enumerate(name_indices_in_seq):
            if i >= len(segment_starts):
                logger.error(
                    f"Mismatch between name_indices_in_seq ({len(name_indices_in_seq)}) and segment_starts ({len(segment_starts)}). Skipping remaining."
                )
                break
            start = segment_starts[i]
            end = name_idx_pos
            name_token_value = active_tokens[name_idx_pos]
            point_sequence_tokens = active_tokens[start:end]

            if not point_sequence_tokens:
                logger.warning(
                    f"Empty point sequence for name token {name_token_value} (index {name_idx_pos}). Skipping."
                )
                continue

            current_polygon_vertices_float = []
            valid_sequence = True
            try:
                for pt_token in point_sequence_tokens:
                    if not (0 <= pt_token < self.NUM_POINT_TOKENS):
                        logger.warning(
                            f"Invalid point token {pt_token} encountered for name token {name_token_value}. Skipping polygon."
                        )
                        valid_sequence = False
                        break
                    disc_x, disc_y = self.token_to_point(pt_token)
                    current_x = disc_x * discretization_factor
                    current_y = disc_y * discretization_factor
                    current_polygon_vertices_float.append((current_x, current_y))

                if valid_sequence:
                    if len(current_polygon_vertices_float) < 3: # A valid polygon for shapely needs at least 3 points
                        logger.warning(
                            f"Reconstructed polygon for name token {name_token_value} has < 3 vertices ({len(current_polygon_vertices_float)}). Skipping."
                        )
                        continue

                    reconstructed_polygons.append(
                        np.round(np.array(current_polygon_vertices_float), 3)
                    )
                    names.append(int(name_token_value - name_start_index))

            except ValueError as e: # From token_to_point
                logger.error(f"ValueError during point token conversion for name {name_token_value}: {e}", exc_info=True)
                continue
            except Exception as e:
                logger.error(
                    f"Error reconstructing polygon for name token {name_token_value}: {e}",
                    exc_info=True,
                )
                continue
        return names, reconstructed_polygons

    def dehumanize(self, human_readable: list[str | int]) -> torch.Tensor: # Changed float to int
        """Converts a human-readable list of strings and integers (point tokens) back into a sequence of tokens."""
        tokens = []
        for item in human_readable:
            # Removed N, W, E, S handling
            if item == "BOS":
                tokens.append(self.BOS_TOKEN)
            # elif item[0] == "<":
            elif re.match(r"<(\d+),(\d+)>", item):
                try:
                    x, y = item[1:-1].split(",")
                    x, y = int(x), int(y)
                    tokens.append(self.point_to_token(x, y))
                except ValueError:
                    raise ValueError(f"Invalid point token: {item}")
            elif isinstance(item, str):
                tokens.append(Mapping.NameToIndex[item] + self.NAME_START_INDEX)
            else:
                raise ValueError(f"Invalid item type: {type(item)} with value {item}")
        return torch.tensor(tokens, dtype=torch.long)

    def humanize(
        self, tokens: torch.Tensor, convert_to_meters: bool = False # convert_to_meters is now less relevant for point tokens
    ) -> list[str | int]: # Changed float to int
        """Converts a sequence of tokens back into a human-readable list of strings and point token integers."""
        human_readable = []
        if tokens.device.type != "cpu":
            tokens = tokens.cpu()
        token_list = tokens.tolist()

        for token_val in token_list:
            if token_val == self.EOS_TOKEN:
                break
            elif token_val == self.PAD_TOKEN:
                continue # Skip padding tokens entirely
            elif token_val == self.BOS_TOKEN:
                human_readable.append("BOS")
            # Removed N, W, E, S handling
            elif token_val >= self.NAME_START_INDEX:
                name_idx = token_val - self.NAME_START_INDEX
                try:
                    # Assuming Mapping.IndexToName provides the direct name string
                    human_readable.append(Mapping.IndexToName[name_idx])
                except (NameError, KeyError, AttributeError) as e: # Added AttributeError
                    print(f"Warning: Could not map name index {name_idx}. Error: {e}") # Changed logger to print
                    human_readable.append(f"NameIdx_{name_idx}")
            elif 0 <= token_val < self.NUM_POINT_TOKENS: # Point token
                if convert_to_meters: # Optionally show coordinates for debugging
                    try:
                        disc_x, disc_y = self.token_to_point(token_val)
                        float_x = disc_x * self.DISCRETIZATION_FACTOR
                        float_y = disc_y * self.DISCRETIZATION_FACTOR
                        human_readable.append(f"<{float_x:.2f},{float_y:.2f}>") # Represent as string if converting
                    except ValueError:
                        human_readable.append(f"<invalid_pt_tok_{token_val}>")
                else:
                    # Output discretized (x,y) for better readability than raw token integer
                    try:
                        disc_x, disc_y = self.token_to_point(token_val)
                        human_readable.append(f"<{disc_x},{disc_y}>")
                    except ValueError:
                        human_readable.append(f"<invalid_pt_tok_{token_val}>")
            else:
                human_readable.append(f"<unk_{token_val}>")
        return human_readable


class SwissDataset(Dataset):
    """
    A dataset class to load and preprocess floorplan data using directional representation.

    Args:
        path (str): Path of the dataset folder containing .pkl files and a CSV index file.
        keys (list[str] | None): Specific sample keys to load. If None, load based on csv_file.
        csv_file (str): Name of the CSV file listing samples.
        reload_sample (bool): If True, clear and regenerate preprocessed cache.
        random_transform (bool): If True, apply random transformations during __getitem__.
        shuffle_spaces (bool): If True, shuffle the order of spaces/doors/boundaries.
        max_samples (int | None): Maximum number of samples to load from the CSV.
    """

    def __init__(
        self,
        path: str,
        keys: list[str] | None = None,
        csv_file: str = "data.csv",
        reload_sample: bool = True,
        augment_data: bool = False,
        random_rotate: bool | None = None,
        random_shuffle: bool | None = None,
        random_mirror: bool | None = None,
        random_stretch: bool | None = None,
        smart_shuffle: bool = False,
        max_samples: int | None = None,
        tokenizer: TokenizationSchema | None = None,
        discretization_factor: float | None = None,
        exclude_door_windows: bool = False,
        exclude_internal: bool = False,
        simple_floorplan: bool = False,
        expected_wall_thickness_mm: int = 450,
    ):
        super().__init__()
        if tokenizer is None:
            assert discretization_factor is not None, (
                "Discretization factor must be provided if tokenization schema is not provided"
            )
            self.schema = TokenizationSchema(
                discretization_factor=discretization_factor,
            )
        else:
            self.schema = tokenizer
            self.schema = tokenizer
        assert self.schema is not None, (
            "Tokenization schema or discretization factor must be provided"
        )
        print("Tokenization schema:")
        pprint(self.schema)
        # Revert to fixed factor like edge.py
        self.factor = 0.001
        self.exclude_door_windows = exclude_door_windows
        self.exclude_internal = exclude_internal
        self.simple_floorplan = simple_floorplan
        self.expected_wall_thickness_mm = expected_wall_thickness_mm
        if (
            self.simple_floorplan
            and not self.exclude_door_windows
            and not self.exclude_internal
        ):
            raise ValueError(
                "Must exclude door windows and internal walls if simple floorplan is True"
            )

        self.augment_data = augment_data
        self.random_rotate = augment_data if random_rotate is None else random_rotate
        self.random_shuffle = augment_data if random_shuffle is None else random_shuffle
        self.random_mirror = augment_data if random_mirror is None else random_mirror
        self.random_stretch = augment_data if random_stretch is None else random_stretch
        self.smart_shuffle = smart_shuffle
        self.successfully_preprocessed = 0
        self.failed_preprocessing = 0

        self.preprocessed_data: dict[str, Any] = {}
        self.path = path
        self.data_index = np.loadtxt(
            os.path.join(self.path, csv_file), dtype="str", delimiter="\t"
        )
        # Use specific keys if provided, otherwise use the index file
        if keys:
            self.data_keys = np.array([k for k in keys if k in self.data_index])
        else:
            self.data_keys = np.array(sorted(self.data_index))

        if max_samples is not None:
            self.data_keys = self.data_keys[:max_samples]

        print(
            f"Loading {len(self.data_keys)} samples from {self.path}.\n"
            f"Augment data: {self.augment_data}\n"
            f"Random rotate: {self.random_rotate}\n"
            f"Random shuffle: {self.random_shuffle}\n"
            f"Random mirror: {self.random_mirror}\n"
            f"Smart shuffle: {self.smart_shuffle}\n"
        )

        self.processed_dir = "processed/"
        if reload_sample and os.path.exists(
            os.path.join(self.path, self.processed_dir)
        ):
            print(
                f"Removing existing processed directory: {os.path.join(self.path, self.processed_dir)}"
            )
            shutil.rmtree(os.path.join(self.path, self.processed_dir))
        os.makedirs(os.path.join(self.path, self.processed_dir), exist_ok=True)

        # Max sequence length (adjust if needed based on new representation)
        self.max_size = 25600  # Original max size in data units (before scaling)

        # Perform initial check and removal of samples causing errors during preprocessing
        self.remove_errored_samples()

    def _discretize(self, value: float) -> int:
        """Discretizes a coordinate value (not length)."""
        # Clamp the value before discretization to avoid exceeding MAX_COORD_LENGTH_VALUE
        # due to floating point inaccuracies or values slightly outside the expected scaled range.
        clamped_value = max(0.0, min(value, self.schema.MAX_COORD_LENGTH_VALUE))
        token = int(round(clamped_value / self.schema.DISCRETIZATION_FACTOR))

        # Ensure token is within the valid discrete range after rounding
        token = max(0, min(token, self.schema.COORD_LENGTH_MAX_VALUE))

        # The check for value > 0 resulting in token = 1 for lengths is not directly applicable here,
        # as this function discretizes individual coordinates.
        # The primary concern is staying within [0, COORD_LENGTH_MAX_VALUE].
        # if token == 0 and value > 1e-9: # A very small positive value might round to 0.
            # This logic might be more relevant for lengths if we want to avoid 0-length segments.
            # For coordinates, (0,0) is a valid point.
            # logger.debug(f"Discretizing small positive value {value} to token {token}, might become 0.")
            # pass # Allow 0 for coordinates

        if not (0 <= token <= self.schema.COORD_LENGTH_MAX_VALUE):
            # This should ideally be prevented by clamping and careful input scaling
            raise ValueError(
                f"Discretized coordinate value {token} (from original {value}, clamped {clamped_value}) is out of range [0, {self.schema.COORD_LENGTH_MAX_VALUE}]"
            )
        return token

    def _undiscretize(self, token: int) -> float:
        """Converts a discrete coordinate token back to a float coordinate value."""
        if not (0 <= token <= self.schema.COORD_LENGTH_MAX_VALUE):
            raise ValueError(
                f"Discrete coordinate token {token} is out of valid range [0, {self.schema.COORD_LENGTH_MAX_VALUE}]"
            )
        return token * self.schema.DISCRETIZATION_FACTOR

    def scale(self, sample: FloorplanData):
        """Scales all coordinates in the sample by the factor."""
        try:
            # Reintroduce assertions consistent with edge.py
            if sample.perimeter is not None:
                if not np.all(sample.perimeter[:, :2] < self.max_size):
                    raise PreprocessingError(
                        f"Perimeter coordinates are too large: max={np.max(sample.perimeter[:, :2])}"
                    )
            if not all(
                y is None or np.all(y < self.max_size) for y in sample.spaces.values()
            ):
                raise PreprocessingError(
                    f"Space coordinates are too large: max={max(np.max(y) for y in sample.spaces.values() if y is not None)}"
                )
            if not all(
                y is None or np.all(y < self.max_size)
                for y in sample.door_windows.values()
            ):
                raise PreprocessingError(
                    f"Door/Window coordinates are too large: max={max(np.max(y) for y in sample.door_windows.values() if y is not None)}"
                )
            if not all(
                y is None or np.all(y < self.max_size)
                for y in sample.boundaries.values()
            ):
                raise PreprocessingError(
                    f"Boundary coordinates are too large: max={max(np.max(y) for y in sample.boundaries.values() if y is not None)}"
                )
            # ... other checks if needed ...

            # Use the fixed self.factor
            scale_factor = self.factor

            if sample.perimeter is not None:
                sample.perimeter = np.array(sample.perimeter) * scale_factor
            sample.spaces = {
                x: np.array(y) * scale_factor
                for x, y in sample.spaces.items()
                if y is not None
            }
            sample.door_windows = {
                x: np.array(y) * scale_factor for x, y in sample.door_windows.items()
            }
            sample.boundaries = {
                x: np.array(y) * scale_factor for x, y in sample.boundaries.items()
            }

        except Exception as e:
            # Wrap potential scaling errors into PreprocessingError
            raise PreprocessingError(
                f"Error during scaling sample {getattr(sample, 'id', 'Unknown')}: {e}"
            )

    def generate_caption(self, sample) -> str:
        """Generates a textual description of the floorplan."""
        desc = ""
        try:
            # Use unscaled coordinates for area calculation if desired, requires loading original
            # Or calculate from scaled, knowing the units
            if sample.perimeter is not None and len(sample.perimeter) >= 3:
                total_area = shapely.Polygon(sample.perimeter[:, :2]).area
            else:
                total_area = 0
                logger.warning(
                    f"Invalid perimeter for area calculation in sample {getattr(sample, 'id', 'Unknown')}"
                )

            num_rooms = len(
                [
                    x
                    for x in sample.spaces
                    if isinstance(x, str)
                    and ("ROOM" in x or "LIVING" in x or "OFFICE" in x)
                ]
            )
            # Adjust units in description based on scaling factor used
            desc += (
                "It is a %d room apartment with a total area of %.1f sq. units. "
                % (
                    num_rooms,
                    total_area,  # Units depend on the scaling factor
                )
            )
            for i, space_coords in sample.spaces.items():
                if (
                    isinstance(space_coords, np.ndarray) and len(space_coords) >= 3
                ):  # Need at least 3 points for a polygon
                    try:
                        space_area = shapely.Polygon(space_coords[:, :2]).area
                        desc += "%s has an area of %.1f sq. units. " % (
                            str(i).split("-")[0],
                            space_area,
                        )
                    except Exception as poly_err:
                        logger.warning(
                            f"Could not calculate area for space {i} in sample {getattr(sample, 'id', 'Unknown')}: {poly_err}"
                        )
                        desc += f"{str(i).split('-')[0]} has invalid geometry. "
                else:
                    desc += f"{str(i).split('-')[0]} has invalid geometry data. "

            # Door connections might need update if structure changed
            door_connections = getattr(sample, "door_connections", {})
            if door_connections:
                for i in door_connections:
                    # Ensure connections exist and are valid indices/keys
                    conn = door_connections[i]
                    if isinstance(conn, (list, tuple)) and len(conn) == 2:
                        desc += "%s is connected to %s. " % (
                            str(conn[0]).split("-")[
                                0
                            ],  # Handle potential non-string types
                            str(conn[1]).split("-")[0],
                        )
                    else:
                        desc += f"Connection {i} is invalid. "

            return desc  # Moved inside try block

        except Exception as e:
            logger.error(
                f"Error generating caption for sample {getattr(sample, 'id', 'Unknown')}: {e}",
                exc_info=True,
            )
            # Return partial description if started, otherwise error message
            return desc if desc else "Error generating description."

    def NameToIndex(self, n: str) -> int:
        """Maps a feature name (e.g., 'ROOM') to an integer index (starting from 0)."""
        # Ensure the mapping exists and returns indices >= 0
        name_part = n.split("-")[0]
        idx = Mapping.NameToIndex.get(name_part)
        if idx is None:
            # Attempt case-insensitive match as fallback?
            for key, val in Mapping.NameToIndex.items():
                if key.lower() == name_part.lower():
                    idx = val
                    logger.warning(
                        f"Name '{name_part}' (from '{n}') not found in Mapping.NameToIndex, using case-insensitive match: {key}"
                    )
                    break
            if idx is None:
                raise ValueError(
                    f"Name '{name_part}' (from '{n}') not found in Mapping.NameToIndex"
                )
        # Assuming Mapping.NameToIndex provides 0-based indices
        return idx

    def IndexToName(self, idx: int) -> str | None:
        """Maps an integer index back to a feature name."""
        # Assuming Mapping.IndexToName exists and maps 0-based indices
        return Mapping.IndexToName.get(idx)

    def convert_polygon_to_point_tokens(
        self, polygon: np.ndarray
    ) -> torch.LongTensor:
        """Converts a rectilinear polygon to a sequence of point tokens [pt_tok1, pt_tok2, ...]."""
        pol = None
        try:
            pol = make_it_rectilinear(polygon) # Expects Nx2 float coordinates
        except PreprocessingError as e:
            raise PreprocessingError(f"make_it_rectilinear failed: {e}")
        except Exception as e:
            raise PreprocessingError(
                f"Unexpected error in make_it_rectilinear for polygon {polygon[:2]}...: {e}"
            )

        if pol is None or len(pol) < 3: # Need at least 3 vertices for a polygon
            raise ValueError(
                f"Polygon is invalid or too small (<3 vertices) after make_it_rectilinear: {pol}"
            )

        # Check coordinate range AFTER potential repositioning/scaling but BEFORE discretization
        if np.any(pol > self.schema.MAX_COORD_LENGTH_VALUE + 1e-6): # Add small tolerance for float comparisons
            max_val = np.max(pol)
            raise ValueError(
                f"Coordinates must be <= {self.schema.MAX_COORD_LENGTH_VALUE:.3f}, but found max value {max_val:.3f} in polygon starting {pol[0]}"
            )
        if np.any(pol < -1e-6): # Check for negative coordinates too
            min_val = np.min(pol)
            raise ValueError(
                f"Coordinates must be >= 0, but found min value {min_val:.3f} in polygon starting {pol[0]}"
            )

        point_tokens = []
        try:
            # Iterate through each vertex of the rectilinear polygon (pol already excludes closing point)
            for i in range(len(pol)):
                x_float, y_float = pol[i, 0], pol[i, 1]

                # Discretize individual coordinates
                # The _discretize method now handles clamping to [0, MAX_COORD_LENGTH_VALUE]
                # and then discretizes to [0, COORD_LENGTH_MAX_VALUE]
                discrete_x = self._discretize(x_float)
                discrete_y = self._discretize(y_float)

                # Convert discretized (x,y) to a single point token
                pt_token = self.schema.point_to_token(discrete_x, discrete_y)
                point_tokens.append(pt_token)

        except ValueError as e: # Catch errors from _discretize or point_to_token
            raise ValueError(f"Error tokenizing point {pol[i]} (float: {x_float},{y_float}): {e}")

        if not point_tokens: # Should have at least 3 point tokens for a polygon
            raise ValueError(
                f"Polygon resulted in zero point tokens (original vertices: {len(polygon)})."
            )
        if len(point_tokens) < 3: # Check again, although initial check on `pol` should catch this
             raise ValueError(
                f"Polygon resulted in too few point tokens (<3): {len(point_tokens)} (original vertices: {len(polygon)})"
            )

        return torch.LongTensor(point_tokens)

    def remove_errored_samples(self):
        """Checks samples in parallel and removes those causing PreprocessingErrors."""
        to_remove = []
        # Limit workers to avoid overwhelming system, ensure at least 1
        num_workers = min(max(1, mp.cpu_count() // 2), 8)  # e.g., max 8 workers
        print(f"Checking {len(self.data_keys)} samples using {num_workers} workers...")

        # Disable multiprocessing temporarily for easier debugging if needed
        # results = [self.check_sample_wrapper(key) for key in self.data_keys]
        with mp.Pool(num_workers) as pool:
            results = pool.map(self.check_sample_wrapper, self.data_keys)

        to_remove = [key for key, error in zip(self.data_keys, results) if error]

        if to_remove:
            print(
                f"Removing {len(to_remove)} out of {len(self.data_keys)} samples due to preprocessing errors."
            )
            # Print some examples of errors
            count = 0
            for key, error in zip(self.data_keys, results):
                if error and count < 10:  # Show more errors
                    print(f"  - Sample {key}: {error}")
                    count += 1
                elif error and count == 10:
                    print("  - ... (more errors)")
                    count += 1  # Prevent printing this line repeatedly
                elif count > 10:
                    break

            original_count = len(self.data_keys)
            self.data_keys = np.array(
                [key for key in self.data_keys if key not in to_remove]
            )
            print(
                f"Remaining samples: {len(self.data_keys)} (removed {original_count - len(self.data_keys)})"
            )
        else:
            print("No samples removed during initial check.")

    def preprocess_sample(
        self,
        key: str,
        rotate: int = 0,
        shuffle: bool = False,
        mirror: bool = False,
        stretch: tuple[float, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Preprocesses a sample and returns a string representation."""
        try:
            sample_obj = load_floorplan(
                f"{self.path}/{key}.pkl",
                simple=self.simple_floorplan,
                exclude_internal=self.exclude_internal,
                expected_wall_thickness_mm=self.expected_wall_thickness_mm,
            )
            transformed_data = self.transform(
                sample_obj,
                rotate=rotate,
                shuffle=shuffle,
                mirror=mirror,
                stretch=stretch,
                smart_shuffle=self.smart_shuffle,
            )
            return self.convert_to_str(transformed_data)
        except Exception as e:
            raise PreprocessingError(f"Error preprocessing sample {key}: {e}")

    def __getitem__(self, index: int) -> str:
        """Returns the preprocessed sample at the given index."""
        if index == 0:
            # demote to debug later
            logger.warning(
                f"Preprocessed {self.successfully_preprocessed} samples successfully and {self.failed_preprocessing} samples failed."
            )
        rotate = random.choice([0, 90, 180, 270]) if self.random_rotate else 0
        shuffle = self.random_shuffle # random.choice([True, False]) if self.random_shuffle else False
        mirror = random.choice([True, False]) if self.random_mirror else False
        stretch = (0.9, 1.1) if self.random_stretch else None

        key = self.data_keys[index]
        try:
            self.successfully_preprocessed += 1
            return self.preprocess_sample(key, rotate, shuffle, mirror, stretch)
        except PreprocessingError:
            self.failed_preprocessing += 1
            self.successfully_preprocessed -= 1
            return self.preprocess_sample(key)

    # ASSUMES THERE IS 1 EXTENDED_BOUNDARY AND 0 OR 1 PERIMETER
    def check_sample_wrapper(self, key: str) -> str | None:
        """Wrapper function for multiprocessing to check a single sample."""
        try:
            _ = self.preprocess_sample(key)
            return None
        except FileNotFoundError:
            # logger.error(f"File not found for sample {key}")
            return f"File not found: {key}.pkl"
        except SequenceTooLongError as e:
            # Return a specific message that remove_errored_samples will recognize
            # logger.warning(str(e)) # Log the warning if desired
            return f"Sequence too long: {e.length} > {e.max_length}"
        except PreprocessingError as e:
            # logger.warning(f"PreprocessingError for sample {key}: {e}")
            return str(e)  # Return error message
        except (ValueError, TypeError, IndexError, AttributeError, KeyError) as e:
            # Catch common data handling errors during processing
            # logger.error(f"Data error checking sample {key}: {e}", exc_info=True)
            return f"Data error: {type(e).__name__}: {e}"
        except Exception as e:
            # Catch other potential errors during loading/transform/conversion
            # logger.error(f"Unexpected error checking sample {key}: {e}", exc_info=True)
            return f"Unexpected error: {type(e).__name__}: {e}"

    @staticmethod
    def _dropout_descriptions(
        descriptions_dict: dict[str, Any],
        top_level_dropout_prob: float = 0.33,
        space_dropout_prob: float = 0.60,
        attribute_dropout_prob: float = 0.50,
    ) -> dict[str, Any]:
        """Applies structured dropout to the descriptions dictionary."""
        processed_descriptions = copy.deepcopy(descriptions_dict)

        # 1. Top-level dropout
        for key in ["total_area", "n_rooms", "space_descriptions"]:
            if random.random() < top_level_dropout_prob:
                if key in processed_descriptions:
                    del processed_descriptions[key]

        # 2. Keys within space_descriptions dropout
        if "space_descriptions" in processed_descriptions:
            space_keys = list(processed_descriptions["space_descriptions"].keys())
            for space_key in space_keys:
                if random.random() < space_dropout_prob:
                    del processed_descriptions["space_descriptions"][space_key]
                else:
                    # 3. Attributes within each surviving space dropout
                    if space_key in processed_descriptions["space_descriptions"]:
                        attribute_keys = list(
                            processed_descriptions["space_descriptions"][space_key].keys()
                        )
                        for attr_key in attribute_keys:
                            if random.random() < attribute_dropout_prob:
                                del processed_descriptions["space_descriptions"][space_key][attr_key]
        return processed_descriptions

    def convert_to_str(self, sample_data: dict) -> dict[str, torch.Tensor]:
        """Converts the preprocessed sample data into a single token sequence."""
        all_tokens: list[int] = []
        added_base_names = set()  # Track base names of added features
        sample_id = sample_data.get("id", "?")  # Get sample ID for error messages
        file_path = sample_data.get(
            "file_path", "?"
        )  # Get file path for error messages
        # --- Prepend BOS token ---
        all_tokens.append(self.schema.BOS_TOKEN)

        # --- Feature Processing Order ---
        # Process extended boundary first IF it exists and is valid
        try:
            ext_boundary_coords = sample_data.get("extended_boundary")
            if (
                ext_boundary_coords is not None
                and isinstance(ext_boundary_coords, np.ndarray)
                and ext_boundary_coords.shape[0] >= 3 # Ensure at least 3 points for a polygon
            ):
                # Use the new conversion method
                poly_point_tokens = self.convert_polygon_to_point_tokens(
                    ext_boundary_coords[:, :2]
                )
                name_idx = self.NameToIndex(
                    "EXTENDED_BOUNDARY"
                )  # Use the actual enum name if available
                name_token = name_idx + self.schema.NAME_START_INDEX
                all_tokens.extend(poly_point_tokens.tolist())
                all_tokens.append(name_token)
                added_base_names.add("EXTENDED_BOUNDARY")  # Record addition
            # else: logger.debug("Skipping extended_boundary (missing or invalid)")
        except (PreprocessingError, ValueError, IndexError, KeyError) as e:
            # logger.warning(
            #     f"Skipping feature 'EXTENDED_BOUNDARY' in sample {sample_id} due to conversion error: {e}"
            # )
            raise PreprocessingError(
                f"Failed to convert 'EXTENDED_BOUNDARY' in sample {sample_id}: {e}"
            )

        # Define the order of other processing features
        feature_dicts_to_process = FEATURES_TO_PROCESS.copy() # Use a copy to modify
        if self.exclude_door_windows:
            feature_dicts_to_process.discard("door_windows")

        # Process perimeter separately IF it exists and is valid
        try:
            perimeter_coords = sample_data.get("perimeter")
            if (
                perimeter_coords is not None
                and isinstance(perimeter_coords, np.ndarray)
                and perimeter_coords.shape[0] >= 3
            ):
                poly_point_tokens = self.convert_polygon_to_point_tokens(
                    perimeter_coords[:, :2]
                )
                name_idx = self.NameToIndex("PERIMETER")
                name_token = name_idx + self.schema.NAME_START_INDEX
                all_tokens.extend(poly_point_tokens.tolist())
                all_tokens.append(name_token)
                added_base_names.add("PERIMETER")  # Record addition
            # else: logger.debug("Skipping perimeter (missing or invalid)")
        except (PreprocessingError, ValueError, IndexError, KeyError) as e:
            # logger.warning(
            #     f"Skipping feature 'PERIMETER' in sample {sample_id} due to conversion error: {e}"
            # )
            raise PreprocessingError(
                f"Failed to convert 'PERIMETER' in sample {sample_id}: {e}"
            )

        # Process feature dictionaries (boundaries, spaces, door_windows)
        for feature_dict_key in feature_dicts_to_process:
            feature_dict = sample_data.get(feature_dict_key, {})
            if not isinstance(feature_dict, dict):
                # logger.warning(
                #     f"Feature '{feature_dict_key}' is not a dictionary in sample {sample_id}. Skipping."
                # )
                # continue
                raise PreprocessingError(
                    f"Feature '{feature_dict_key}' is not a dictionary in sample {sample_id}."
                )

            item_list = list(feature_dict.items())
            if not self.random_shuffle:  # Only sort if NOT shuffling
                # Sort by name str 'ROOM-1' etc. Handle non-string keys gracefully.
                try:
                    item_list = sorted(item_list, key=lambda item: str(item[0]))
                except Exception as sort_err:
                    logger.warning(
                        f"Could not sort feature dict {feature_dict_key} by key name: {sort_err}. Proceeding unsorted."
                    )
            # else: If shuffling, iterate in the order provided by the dictionary (already shuffled)

            for name, coords in item_list:
                name_str = str(name)
                base_name = name_str.split("-")[
                    0
                ].upper()  # Use uppercase for consistency

                # --- Skip if this base feature name was already added --- #
                if base_name in added_base_names:
                    # logger.debug(f"Skipping feature '{name_str}' in sample {sample_id} because base name '{base_name}' was already added.")
                    continue
                # -------------------------------------------------------- #

                try:
                    # Ensure coords are numpy array for conversion
                    if not isinstance(coords, np.ndarray):
                        coords = np.array(coords)

                    # Need at least 2 points for a polyline (doors/windows might be lines)
                    # Need at least 3 points for polygon conversion (checked in convert_polygon...)
                    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
                        # logger.warning(
                        #     f"Skipping feature '{name}' in sample {sample_id} due to insufficient coordinate data: shape={coords.shape}"
                        # )
                        # continue
                        raise PreprocessingError(
                            f"Feature '{name_str}' in sample {sample_id} has insufficient coordinate data: shape={coords.shape}"
                        )

                    # Convert polygon to directional string
                    poly_tokens = self.convert_polygon_to_point_tokens(coords[:, :2]) # Use new method

                    # Get name index and add offset
                    name_idx = self.NameToIndex(name_str)
                    name_token = name_idx + self.schema.NAME_START_INDEX

                    # Append tokens: [anchor_x, anchor_y, dir1, len1, ..., name_token]
                    all_tokens.extend(poly_tokens.tolist())
                    all_tokens.append(name_token)

                except (PreprocessingError, ValueError, IndexError, KeyError) as e:
                    # logger.warning(
                    #     f"Skipping feature '{name}' in sample {sample_id} due to conversion error: {e}"
                    # )
                    # continue  # Skip this feature if conversion fails
                    raise PreprocessingError(
                        f"Failed to convert feature '{name_str}' in sample {sample_id}: {e}"
                    )

        # Add EOS token
        all_tokens.append(self.schema.EOS_TOKEN)

        # --- Check sequence length BEFORE padding ---
        seq_len = len(all_tokens)
        if seq_len > self.schema.MAX_SEQ_LEN:
            # Instead of truncating, raise the specific error
            raise SequenceTooLongError(
                key=sample_id,
                length=seq_len,
                max_length=self.schema.MAX_SEQ_LEN,
            )
            # logger.warning(f"Sequence length {seq_len} exceeds max_length {self.schema.MAX_SEQ_LEN} for sample {sample_id}. Truncating.")
            # all_tokens = all_tokens[:self.schema.MAX_SEQ_LEN]
            # if all_tokens[-1] != self.schema.EOS_TOKEN:
            #     all_tokens[-1] = self.schema.EOS_TOKEN
            # seq_len = self.schema.MAX_SEQ_LEN
        elif seq_len == 1:  # Only EOS token means no features were converted
            raise PreprocessingError(
                f"Sample {sample_id} resulted in an empty sequence (only EOS token)."
            )
        # --- End length check ---

        pad_length = self.schema.MAX_SEQ_LEN - seq_len
        padded_tokens = all_tokens + [self.schema.PAD_TOKEN] * pad_length

        # Create tensors
        input_ids = torch.LongTensor(padded_tokens)
        # Attention mask: 1 for real tokens (including EOS), 0 for padding
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        attention_mask[:seq_len] = True
        # Labels are typically the same as input_ids for auto-regressive models
        labels = input_ids.clone()

        # Generate descriptions
        descriptions_dict = self.generate_descriptions(all_tokens)

        # Apply structured dropout by calling the new static method
        processed_descriptions = self._dropout_descriptions(descriptions_dict)
        
        descriptions_json_str = json.dumps(processed_descriptions)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "id": sample_id,  # Pass ID for debugging collate_fn issues
            "file_path": file_path,  # For debugging
            "descriptions_text": descriptions_json_str, # Dropout applied JSON string
            "original_descriptions_dict": descriptions_dict, # Original, unmodified dict
        }
        result["attention_mask"][len(all_tokens) :] = 0  # Mask out padding

        return result

    def generate_descriptions(self, all_tokens: list[int]) -> dict[str, Any]:
        """Generates descriptions for the given token sequence."""
        # This method needs significant rework for point token sequences
        # It now needs to reconstruct polygons from sequences of point tokens

        space_descriptions = {}
        name_counter = defaultdict(int)
        current_point_token_sequence: list[int] = []
        # sample_total_area = 0 # This will be derived from PERIMETER space
        # n_rooms = 0 # This will be derived from number of ROOM-like spaces

        # Iterate through tokens, skipping BOS and stopping at EOS/PAD
        # Assuming active_tokens does not include EOS/PAD but might include BOS
        # For safety, let's filter BOS here if present and handle EOS inside loop

        iter_tokens = iter(all_tokens)
        try:
            first_token = next(iter_tokens)
            if first_token != self.schema.BOS_TOKEN:
                # If BOS is not the first, something is off or the input `all_tokens` was pre-filtered
                # Put it back and process, or raise an error.
                # For now, let's assume `all_tokens` starts with BOS and we consume it.
                # If not, the first token might be a point or name, handle accordingly.
                # This logic depends on how `all_tokens` is passed. Assuming it includes BOS.
                logger.warning("generate_descriptions expected BOS_TOKEN as first token.")
                # Re-add if it wasn't BOS, to process it as a point/name token
                # This is a bit tricky. For now, let's assume BOS is always first and we skipped it.
                # Let's proceed assuming it was BOS and we skipped it.
                if first_token < self.schema.NAME_START_INDEX: # It was a point token
                    current_point_token_sequence.append(first_token)
                # else it might be a name token, which is unusual to be first after BOS typically.

        except StopIteration:
            return { # Empty sequence after potential BOS
                "total_area": 0,
                "n_rooms": 0,
                "space_descriptions": space_descriptions,
            }

        for token_val in iter_tokens: # Process remaining tokens
            if token_val == self.schema.EOS_TOKEN:
                # Process any pending polygon before breaking if EOS is encountered
                # This typically shouldn't happen if name token is last before EOS
                if current_point_token_sequence:
                    logger.warning("EOS token encountered with pending point tokens. Discarding them as no name token followed.")
                    current_point_token_sequence = [] # Clear unfinished polygon
                break # End of sequence

            if token_val >= self.schema.NAME_START_INDEX: # It's a name token
                original_name_from_schema = self.IndexToName(token_val - self.schema.NAME_START_INDEX)
                if original_name_from_schema is None:
                    logger.warning(f"Could not map name index {token_val - self.schema.NAME_START_INDEX} to name. Skipping feature.")
                    current_point_token_sequence = [] # Discard points for unknown name
                    continue

                current_name = original_name_from_schema
                if current_name not in ["EXTENDED_BOUNDARY", "PERIMETER"]:
                    name_counter[current_name] += 1
                    current_name = f"{current_name}-{name_counter[current_name]}"

                if not current_point_token_sequence:
                    logger.warning(f"Name token '{current_name}' encountered without preceding point tokens. Skipping feature.")
                    continue # No points for this name

                # Reconstruct polygon from current_point_token_sequence
                polygon_vertices_float = []
                valid_poly = True
                for pt_token in current_point_token_sequence:
                    try:
                        disc_x, disc_y = self.schema.token_to_point(pt_token)
                        # No need to multiply by discretization_factor here if we use discrete coords for shapely polygon for area calc
                        # Actually, for area, we DO need float coordinates
                        float_x = disc_x * self.schema.DISCRETIZATION_FACTOR
                        float_y = disc_y * self.schema.DISCRETIZATION_FACTOR
                        polygon_vertices_float.append((float_x, float_y))
                    except ValueError as e:
                        logger.warning(f"Invalid point token {pt_token} for name '{current_name}' during description generation: {e}. Skipping polygon.")
                        valid_poly = False
                        break
                
                if valid_poly and len(polygon_vertices_float) >= 3:
                    try:
                        polygon = shapely.Polygon(polygon_vertices_float)
                        area = round(polygon.area, 2) # Use more precision for area
                        # Complexity: number of vertices. The `make_it_rectilinear` ensures no redundant closing point.
                        complexity = len(polygon_vertices_float)
                        space_descriptions[current_name] = {
                            "area": area,
                            "complexity": complexity, # Number of vertices
                        }
                    except Exception as e:
                        logger.warning(f"Shapely error for '{current_name}' with {len(polygon_vertices_float)} vertices: {e}. Skipping.")
                elif valid_poly: # < 3 vertices
                    logger.warning(f"Feature '{current_name}' has too few vertices ({len(polygon_vertices_float)}) for description. Skipping.")

                current_point_token_sequence = [] # Reset for the next feature

            elif 0 <= token_val < self.schema.NUM_POINT_TOKENS: # It's a point token
                current_point_token_sequence.append(token_val)
            else: # Should be PAD or other unknown, which should ideally be filtered before this function
                logger.warning(f"Unexpected token {token_val} in generate_descriptions. Skipping.")
                # If there were pending points, they might be orphaned if this isn't a name token.
                # current_point_token_sequence = [] # Consider clearing if an unexpected token breaks a sequence

        # Final calculations for top-level descriptions
        # Revert to simpler calculation assuming PERIMETER is always present
        # and EXTENDED_BOUNDARY is also present, so n_rooms is total spaces - 2.
        # Error handling during tokenization in convert_to_str should prevent
        # reaching this point if PERIMETER or EXTENDED_BOUNDARY failed.

        if "PERIMETER" not in space_descriptions:
             # This case should ideally not be reached if preprocessing is robust.
             # Log a warning and attempt to calculate area from extended_boundary or set to 0.
            logger.warning("PERIMETER key missing in space_descriptions for description generation.")
            if "EXTENDED_BOUNDARY" in space_descriptions:
                total_area = space_descriptions["EXTENDED_BOUNDARY"].get("area", 0)
                logger.warning("Using EXTENDED_BOUNDARY area as fallback for total_area.")
            else:
                total_area = 0
                logger.warning("Neither PERIMETER nor EXTENDED_BOUNDARY area found. Setting total_area to 0.")
        else:
            total_area = space_descriptions["PERIMETER"].get("area", 0)


        # n_rooms is the count of all entries in space_descriptions minus PERIMETER and EXTENDED_BOUNDARY
        # This assumes both are always present in space_descriptions if processing was successful.
        # If they are not, len(space_descriptions) would be smaller.
        # Count only actual rooms if specific naming convention allows,
        # otherwise, it's total items minus the two special boundaries.
        
        # Original simpler logic assumed PERIMETER and EXTENDED_BOUNDARY are always there.
        # Let's refine to count based on actual keys present, subtracting known non-room items.
        non_room_keys = {"PERIMETER", "EXTENDED_BOUNDARY"}
        n_rooms = len(space_descriptions) - len(non_room_keys)

        return {
            "total_area": total_area,
            "n_rooms": n_rooms,
            "space_descriptions": space_descriptions,
        }

    def __len__(self):
        return len(self.data_keys)
