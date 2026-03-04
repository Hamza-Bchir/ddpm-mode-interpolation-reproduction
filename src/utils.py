import torch
import torch.nn.functional as F
import random
import numpy as np
import math
from torch.utils.data import Dataset
import itertools
from typing import Optional, Tuple, Union, List
from PIL import Image, ImageDraw



def hallucination_metric(
    xhat0_traj: torch.Tensor,
    *,
    time_dim: int = 1,
    reduce: str = "mean",
    keepdim: bool = False,
) -> torch.Tensor:
    """
    Hallucination metric from the paper (Eq. 4): variance of x̂0 trajectory over timesteps.

    Args:
        xhat0_traj:
            Predicted x0 trajectory with shape (B, T, ...) by default.
            It can be (B,T,D), (B,T,C,H,W), etc. Any shape is fine as long as one dim is time.
        time_dim:
            Which dimension corresponds to timesteps T.
        reduce:
            How to reduce across the remaining (non-batch, non-time) dims to get one score per sample:
            - "mean": mean over feature/image dims (matches paper wording: var per-dim then mean).
            - "sum": sum over feature dims.
            - "none": return the per-element variance map/tensor (shape (B, ...)).
        keepdim:
            If reduce != "none", whether to keep reduced dims (PyTorch keepdim semantics).

    Returns:
        Tensor of shape:
            - (B,) if reduce in {"mean","sum"} and keepdim=False
            - (B,1,...) if keepdim=True (exact kept shape depends)
            - (B, ...) if reduce=="none"
    """
    if xhat0_traj.dim() < 2:
        raise ValueError("xhat0_traj must have at least batch and time dimensions.")

    # Move time dimension to dim=1: (B, T, ...)
    if time_dim != 1:
        x = xhat0_traj.movedim(time_dim, 1)
    else:
        x = xhat0_traj

    if x.size(1) < 2:
        raise ValueError("Need at least 2 timesteps to compute a variance.")

    # Population variance over time (divide by T), matching Eq.(4) which uses 1/|T2-T1| * sum(...)
    # torch.var(unbiased=False) is population variance.
    var_t = x.var(dim=1, unbiased=False)  # shape (B, ...)

    if reduce == "none":
        return var_t

    # Reduce over all remaining dims except batch
    reduce_dims = tuple(range(1, var_t.dim()))
    if len(reduce_dims) == 0:
        # var_t is already (B,)
        return var_t

    if reduce == "mean":
        return var_t.mean(dim=reduce_dims, keepdim=keepdim)
    if reduce == "sum":
        return var_t.sum(dim=reduce_dims, keepdim=keepdim)

    raise ValueError(f"Unknown reduce='{reduce}'. Use 'mean', 'sum', or 'none'.")

def set_seed(seed: int = 0) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def sinusoidal_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10_000
) -> torch.Tensor:
    """
    Compute sinusoidal timestep embeddings as commonly used in diffusion models.
    """
    if t.dim() != 1:
        raise ValueError(f"`t` must be 1D of shape (B,), got {tuple(t.shape)}")

    if dim < 2:
        raise ValueError(f"`dim` must be >= 2, got {dim}")

    device = t.device
    B = t.shape[0]

    half_dim = dim // 2
    if half_dim < 1:
        raise ValueError(f"`dim` must allow at least one sine/cosine pair, got {dim}")

    # Frequencies: exp(-log(max_period) * i / (half_dim - 1))
    # Handles half_dim == 1 safely
    if half_dim == 1:
        freqs = torch.ones(1, device=device, dtype=torch.float32)
    else:
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / (half_dim - 1)
        )

    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (B, 2*half_dim)

    # If dim is odd, pad one dimension
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb



class GaussianDataset(Dataset):
    """
    Dataset representing a Gaussian distribution or a mixture of Gaussians.

    This dataset generates synthetic samples drawn from either:
        - A single Gaussian distribution
        - A mixture of multiple Gaussian components with uniform weights

    It supports:
        - Arbitrary number of mixture components
        - Configurable means and standard deviations
        - Shared standard deviation across components
        - Preset configurations for common 1D and 2D cases

    Preset Behavior (if preset=True):
        - dim=1:
            * 3 components
            * Means: [[1.0], [2.0], [3.0]]
            * Shared std: 0.05

        - dim=2:
            * 25 components arranged on a 5×5 equally spaced grid
            * Grid bounds controlled by grid_min_2d and grid_max_2d
            * Shared std controlled by std_2d

    Each sample consists of:
        - x: data point drawn from selected Gaussian component
        - y: integer component index

    Attributes:
        n (int):
            Total number of samples.

        dim (int):
            Dimensionality of each data point.

        n_components (int):
            Number of Gaussian mixture components.

        means (torch.Tensor):
            Tensor of shape (n_components, dim) containing component means.

        stds (torch.Tensor):
            Tensor of shape (n_components, dim) containing component standard deviations.
    """

    def __init__(
        self,
        n: int,
        dim: int,
        n_components: int | None = None,
        means=None,
        stds=None,
        shared_std: bool = True,
        preset: bool = True,
        grid_size_2d: int = 5,
        grid_min_2d: float = -2.0,
        grid_max_2d: float = 2.0,
        std_2d: float = 0.05,
    ) -> None:
        """
        Initialize the GaussianDataset.

        Args:
            n (int):
                Total number of samples to generate.

            dim (int):
                Dimensionality of each sample.

            n_components (int | None, optional):
                Number of Gaussian mixture components.
                Required if preset=False and means are not provided.

            means (array-like or torch.Tensor, optional):
                Mean vectors for each component.
                Expected shape: (n_components, dim).

            stds (float, int, array-like, or torch.Tensor, optional):
                Standard deviations for each component.
                If shared_std=True, must be a scalar.
                Otherwise, must match shape (n_components, dim).

            shared_std (bool, optional):
                If True, a single scalar std is shared across all components.
                If False, each component may have its own std.
                Default is True.

            preset (bool, optional):
                If True, automatically constructs predefined mixtures for dim=1 or dim=2.
                Default is True.

            grid_size_2d (int, optional):
                Number of grid points per axis for 2D preset.
                Default is 5.

            grid_min_2d (float, optional):
                Minimum coordinate value for 2D grid preset.
                Default is -2.0.

            grid_max_2d (float, optional):
                Maximum coordinate value for 2D grid preset.
                Default is 2.0.

            std_2d (float, optional):
                Shared standard deviation used in 2D preset.
                Default is 0.2.
        """
        super().__init__()
        self.n = int(n)
        self.dim = int(dim)

        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")

        # ---- presets
        if preset:
            if self.dim == 1:
                n_components = 3
                means = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
                stds = 0.05
                shared_std = True
            elif self.dim == 2:
                coords = torch.linspace(grid_min_2d, grid_max_2d, grid_size_2d)
                grid = list(itertools.product(coords.tolist(), coords.tolist()))
                means = torch.tensor(grid, dtype=torch.float32)  # (grid_size^2, 2)
                n_components = int(means.size(0))
                stds = float(std_2d)
                shared_std = True
            else:
                raise ValueError("preset=True is only implemented for dim=1 or dim=2")

        if means is None:
            raise ValueError("means must be provided when preset=False")

        means = torch.as_tensor(means, dtype=torch.float32)
        if means.dim() == 1:
            means = means.unsqueeze(1)  # (K,) -> (K,1)

        if means.dim() != 2:
            raise ValueError(f"means must be 2D after conversion, got shape {tuple(means.shape)}")

        if means.shape[1] != self.dim:
            raise ValueError(
                f"Means dimension mismatch: expected dim={self.dim}, got {means.shape[1]}"
            )

        if n_components is not None and int(n_components) != means.shape[0]:
            raise ValueError(
                f"n_components={n_components} but means has {means.shape[0]} components"
            )

        self.n_components = int(means.shape[0])
        self.means = means

        # ---- stds handling
        if shared_std:
            if not isinstance(stds, (float, int)):
                raise ValueError("When shared_std=True, stds must be a scalar float/int")
            stds_t = torch.full(
                (self.n_components, self.dim),
                float(stds),
                dtype=torch.float32,
            )
        else:
            if stds is None:
                raise ValueError("stds must be provided when shared_std=False")
            stds_t = torch.as_tensor(stds, dtype=torch.float32)
            if stds_t.dim() == 1:
                # (K,) or (K,1) style -> expand to (K, dim)
                stds_t = stds_t.unsqueeze(1).repeat(1, self.dim)
            if stds_t.shape != self.means.shape:
                raise ValueError(
                    f"stds must match means shape {tuple(self.means.shape)} when shared_std=False, "
                    f"got {tuple(stds_t.shape)}"
                )

        self.stds = stds_t

        # ---- sample mixture with uniform weights
        comp_ids = torch.randint(0, self.n_components, (self.n,), dtype=torch.long)
        noise = torch.randn(self.n, self.dim, dtype=torch.float32)
        self.x = self.means[comp_ids] + self.stds[comp_ids] * noise
        self.y = comp_ids  # component id

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.n

    def __getitem__(self, idx: int):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            (x, y) where:
                x: Tensor of shape (dim,)
                y: Long tensor scalar (component index)
        """
        return self.x[idx], self.y[idx]
    
import math
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class HandShapesDataset(Dataset):
    """
    Synthetic grayscale dataset of a simple "hand-like" shape.

    Each sample is a single-channel image (H,W) in {0,1} (or [0,1] if soft edges),
    consisting of:
      - a filled circle ("palm")
      - 5 thick line segments ("fingers") emanating from the palm border
      - a thumb separated in angle to make global orientation identifiable

    This version supports:
      1) Fixed (or ranged) finger thickness via `finger_width`.
      2) Discrete orientation support via `orientation_mode="binned"`.
      3) Discrete *positional support by regions* via `position_mode="columns"`:
         split image into vertical columns (default 3). Allow sampling centers only in
         specified columns (e.g., left+right), making the middle column out-of-support.

    Determinism:
      - Generation is deterministic per index: RNG uses seed + idx.

    Practical defaults for sharp "support":
      - image_size=64
      - position_mode="columns", n_columns=3, allowed_columns=("left","right")
      - orientation_mode="binned", n_orientation_bins=8 or 12
      - center_jitter=0
      - palm_radius_range=(12,12), finger_len_range=(18,18), finger_width=3
    """

    def __init__(
        self,
        n: int,
        image_size: int = 64,
        antialias: int = 4,
        return_label: bool = False,
        n_orientation_bins: int = 8,
        # Appearance
        bg_value: int = 0,
        fg_value: int = 255,
        binarize: bool = True,
        binarize_thresh: float = 0.5,
        # Geometry
        palm_radius_range: Tuple[int, int] = (10, 14),
        finger_len_range: Tuple[int, int] = (14, 22),

        # Thickness control: int=fixed, (min,max)=random per sample
        finger_width: Union[int, Tuple[int, int]] = 3,
        thumb_width_scale: float = 1.05,  # set 1.0 for identical thickness

        spread_deg: float = 70.0,
        thumb_angle_offset_deg: float = 55.0,
        thumb_len_scale: float = 0.85,

        # Orientation control
        orientation_mode: str = "continuous",  # {"continuous","binned"}

        # Position control
        position_mode: str = "random",  # {"random","columns"}
        center_jitter: int = 6,         # used in random mode (and optional extra jitter in columns mode)

        # "columns" mode params
        n_columns: int = 3,
        allowed_columns: Sequence[str] = ("left", "right"),  # any of {"left","middle","right"} for n_columns=3
        y_range: Optional[Tuple[int, int]] = None,           # center-y range in output px; None => auto-safe
        enforce_in_bounds: bool = True,                      # try to keep entire shape inside image

        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.image_size = int(image_size)
        self.antialias = int(antialias)
        self.return_label = bool(return_label)
        self.n_orientation_bins = int(n_orientation_bins)

        self.bg_value = int(bg_value)
        self.fg_value = int(fg_value)
        self.binarize = bool(binarize)
        self.binarize_thresh = float(binarize_thresh)

        self.palm_radius_range = tuple(map(int, palm_radius_range))
        self.finger_len_range = tuple(map(int, finger_len_range))

        self.finger_width = finger_width
        self.thumb_width_scale = float(thumb_width_scale)

        self.spread_deg = float(spread_deg)
        self.thumb_angle_offset_deg = float(thumb_angle_offset_deg)
        self.thumb_len_scale = float(thumb_len_scale)

        self.orientation_mode = str(orientation_mode)

        self.position_mode = str(position_mode)
        self.center_jitter = int(center_jitter)

        self.n_columns = int(n_columns)
        self.allowed_columns = tuple(allowed_columns)
        self.y_range = y_range
        self.enforce_in_bounds = bool(enforce_in_bounds)

        self.seed = int(seed)
        self.dtype = dtype

        # ---- checks
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.antialias <= 0:
            raise ValueError(f"antialias must be positive, got {self.antialias}")
        if self.n_orientation_bins <= 0:
            raise ValueError(f"n_orientation_bins must be positive, got {self.n_orientation_bins}")
        if self.orientation_mode not in {"continuous", "binned"}:
            raise ValueError("orientation_mode must be one of {'continuous','binned'}")
        if not (0.0 < self.binarize_thresh < 1.0):
            raise ValueError("binarize_thresh should be in (0,1)")
        if self.position_mode not in {"random", "columns"}:
            raise ValueError("position_mode must be one of {'random','columns'}")
        if self.n_columns <= 0:
            raise ValueError("n_columns must be positive")

        if self.position_mode == "columns":
            # map allowed column names -> indices
            valid_names = {"left", "middle", "right"} if self.n_columns == 3 else None
            if valid_names is not None:
                for name in self.allowed_columns:
                    if name not in valid_names:
                        raise ValueError(f"allowed_columns contains '{name}' but valid are {sorted(valid_names)}")

            if len(self.allowed_columns) == 0:
                raise ValueError("allowed_columns must be non-empty when position_mode='columns'")

    def __len__(self) -> int:
        return self.n

    @staticmethod
    def _pol2cart(r: float, theta: float) -> Tuple[float, float]:
        return r * math.cos(theta), r * math.sin(theta)

    def _sample_orientation(self, rng: np.random.Generator) -> Tuple[float, int]:
        """
        Returns:
            ori: float angle in radians
            y: orientation bin index in [0, n_bins-1]
        """
        if self.orientation_mode == "continuous":
            ori = float(rng.uniform(0.0, 2.0 * math.pi))
            y = int((ori / (2.0 * math.pi)) * self.n_orientation_bins) % self.n_orientation_bins
            return ori, y

        y = int(rng.integers(0, self.n_orientation_bins))
        ori = (2.0 * math.pi) * (y / self.n_orientation_bins)
        return float(ori), y

    def _sample_base_width(self, rng: np.random.Generator) -> int:
        """
        Returns a stroke width in output pixels (not AA-scaled).
        - int: fixed width
        - tuple: random width in [min,max]
        """
        if isinstance(self.finger_width, int):
            return int(self.finger_width)

        if (
            isinstance(self.finger_width, tuple)
            and len(self.finger_width) == 2
            and all(isinstance(v, int) for v in self.finger_width)
        ):
            wmin, wmax = self.finger_width
            if wmin <= 0 or wmax <= 0 or wmin > wmax:
                raise ValueError(f"Invalid finger_width range: {self.finger_width}")
            return int(rng.integers(wmin, wmax + 1))

        raise TypeError("finger_width must be int or (min:int, max:int)")

    def _sample_center(self, rng: np.random.Generator, max_extent: int) -> Tuple[int, int]:
        """
        Sample center (cx, cy) in OUTPUT pixels.
        max_extent: approximate max radius from center to any drawn pixel (in output px).
        """
        W = H = self.image_size

        if self.position_mode == "random":
            cx = W // 2 + int(rng.integers(-self.center_jitter, self.center_jitter + 1))
            cy = H // 2 + int(rng.integers(-self.center_jitter, self.center_jitter + 1))
            return cx, cy

        # columns mode
        col_w = W / self.n_columns

        def name_to_col_idx(name: str) -> int:
            if self.n_columns == 3:
                return {"left": 0, "middle": 1, "right": 2}[name]
            # For n_columns != 3, allow numeric strings "0","1",... or "col0",... (optional)
            # Here: simplest: require ints encoded as strings.
            return int(name)

        allowed_idxs = [name_to_col_idx(c) for c in self.allowed_columns]

        col_idx = int(rng.choice(allowed_idxs))

        x0 = int(math.floor(col_idx * col_w))
        x1 = int(math.floor((col_idx + 1) * col_w)) - 1
        x1 = max(x1, x0)

        if self.enforce_in_bounds:
            x0 = max(x0 + max_extent, 0)
            x1 = min(x1 - max_extent, W - 1)
            if x1 < x0:
                # If the column is too narrow for max_extent, fall back to safest position in that column
                x0 = int(math.floor(col_idx * col_w))
                x1 = int(math.floor((col_idx + 1) * col_w)) - 1
                cx = int((x0 + x1) / 2)
            else:
                cx = int(rng.integers(x0, x1 + 1))
        else:
            cx = int(rng.integers(x0, x1 + 1))

        # y range
        if self.y_range is not None:
            y0, y1 = self.y_range
            y0, y1 = int(y0), int(y1)
        else:
            y0, y1 = 0, H - 1

        if self.enforce_in_bounds:
            y0 = max(y0 + max_extent, 0)
            y1 = min(y1 - max_extent, H - 1)
            if y1 < y0:
                cy = H // 2
            else:
                cy = int(rng.integers(y0, y1 + 1))
        else:
            cy = int(rng.integers(y0, y1 + 1))

        # optional additional jitter even in columns mode
        if self.center_jitter > 0:
            cx += int(rng.integers(-self.center_jitter, self.center_jitter + 1))
            cy += int(rng.integers(-self.center_jitter, self.center_jitter + 1))

        return cx, cy

    def _draw_one(self, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
        """
        Draw a single sample as a numpy array in [0,1], shape (H,W).
        """
        size = self.image_size
        aa = self.antialias
        S = size * aa

        # sample geometry in OUTPUT pixels
        palm_r = int(rng.integers(self.palm_radius_range[0], self.palm_radius_range[1] + 1))
        base_len = int(rng.integers(self.finger_len_range[0], self.finger_len_range[1] + 1))
        base_w = self._sample_base_width(rng)

        # approximate max extent from center for in-bounds sampling (very conservative)
        # palm radius + finger length + stroke padding
        max_extent = palm_r + base_len + max(2, base_w)

        cx0, cy0 = self._sample_center(rng, max_extent=max_extent)

        # convert to AA coordinates
        cx = int(cx0 * aa)
        cy = int(cy0 * aa)
        r0 = int(palm_r * aa)

        img = Image.new("L", (S, S), color=self.bg_value)
        draw = ImageDraw.Draw(img)

        # palm
        draw.ellipse((cx - r0, cy - r0, cx + r0, cy + r0), fill=self.fg_value)

        # orientation + angles
        ori, y = self._sample_orientation(rng)

        spread = math.radians(self.spread_deg)
        offsets = np.linspace(-spread / 2.0, spread / 2.0, 4)

        thumb_offset = math.radians(self.thumb_angle_offset_deg)
        thumb_side = 1.0 if rng.random() < 0.5 else -1.0
        thumb_angle = ori + thumb_side * (spread / 2.0 + thumb_offset)

        angles = [thumb_angle] + [ori + float(o) for o in offsets]

        # widths in AA coordinates (fixed across non-thumb fingers)
        base_w_aa = max(1, int(base_w * aa))
        thumb_w_aa = max(1, int(base_w_aa * self.thumb_width_scale))

        # lengths in AA coordinates
        base_len_aa = int(base_len * aa)
        thumb_len_aa = int(base_len_aa * self.thumb_len_scale)

        # draw 5 fingers
        for i, ang in enumerate(angles):
            start_r = r0 - int(0.5 * aa)  # ensure connectivity
            sx_off, sy_off = self._pol2cart(start_r, ang)
            sx, sy = cx + sx_off, cy + sy_off

            if i == 0:  # thumb
                L = thumb_len_aa
                w = thumb_w_aa
            else:
                L = base_len_aa
                w = base_w_aa

            ex_off, ey_off = self._pol2cart(start_r + L, ang)
            ex, ey = cx + ex_off, cy + ey_off

            draw.line((sx, sy, ex, ey), fill=self.fg_value, width=w)

            tip_r = max(1, w // 2)
            draw.ellipse((ex - tip_r, ey - tip_r, ex + tip_r, ey + tip_r), fill=self.fg_value)

        # downsample
        if aa != 1:
            img = img.resize((size, size), resample=Image.Resampling.LANCZOS)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.binarize:
            arr = (arr >= self.binarize_thresh).astype(np.float32)
        return arr, y

    def __getitem__(self, idx: int):
        """
        Returns:
            If return_label=False:
                x: Tensor (1,H,W) in {0,1} (or [0,1] if binarize=False)
            If return_label=True:
                (x, y): y is orientation bin in [0, n_bins-1]
        """
        if not (0 <= idx < self.n):
            raise IndexError(f"idx out of range: {idx}")

        rng = np.random.default_rng(self.seed + int(idx))
        arr, y = self._draw_one(rng)
        x = torch.from_numpy(arr).to(dtype=self.dtype).unsqueeze(0)  # (1,H,W)

        if self.return_label:
            return x, torch.tensor(y, dtype=torch.long)
        return x