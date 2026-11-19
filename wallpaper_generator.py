#!/usr/bin/env python3
"""Pseudo-random pixel art wallpaper generator (8K height). MVP.

- Output height is 8K (4320 px); width follows the chosen aspect ratio.
- Every format uses 48 px cells and 90 rows, so pixels scale cleanly.
- Each dot is a square smaller than its cell; the gap between dots is the padding.
- Pseudo-random: seeded walkers light up cells, forming organic clusters.
  Same seed + same format + same color + same padding = same image.
- Shading: cell brightness depends on how many lit neighbors it has, so
  group centers are brightest and edges dim, with some random variation.
- Rainbow mode (--rainbow or --color rainbow): each dot's hue follows a gradient
  across the image, with a little random hue variation. Start hue and gradient
  direction come from the seed (so each seed looks different); the gradient covers
  only part of the hue circle, so it does not loop back to its first color.
  Override with --hue-start, --hue-span and --hue-angle.
- Multi-color mode (--color "#FF0000,#0000FF", up to 4 colors): each color is placed
  at a random point of the image and the colors blend smoothly between those points.
  Positions come from --color-seed (default: the main seed). One color = solid color.
- The seed is chosen by the user and must be a non-negative integer (the interactive
  menu can also draw a random one for you).

Usage:
    python3 wallpaper_generator.py --seed 6
    python3 wallpaper_generator.py --seed 6 --format 21:9 --color "#FF6A00"
    python3 wallpaper_generator.py --seed 6 --rainbow
    python3 wallpaper_generator.py -i        # interactive menu
"""

import argparse
import json
import random
import re
import secrets
import shutil
import string
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

# "8K" means 4320 px height. Width comes from the aspect ratio.
# All formats use 48 px cells, so width and height are exact multiples of the cell.
CELL_SIZE = 48
HEIGHT_8K = 4320
FORMATS = {
    "16:9": (7680, HEIGHT_8K, CELL_SIZE),   # 160 x 90 cells
    "16:10": (6912, HEIGHT_8K, CELL_SIZE),  # 144 x 90 cells
    "21:9": (10080, HEIGHT_8K, CELL_SIZE),  # 210 x 90 cells
    "32:9": (15360, HEIGHT_8K, CELL_SIZE),  # 320 x 90 cells
    "4:3": (5760, HEIGHT_8K, CELL_SIZE),    # 120 x 90 cells
    "3:2": (6480, HEIGHT_8K, CELL_SIZE),    # 135 x 90 cells
    "1:1": (4320, HEIGHT_8K, CELL_SIZE),    # 90 x 90 cells
}
DEFAULT_FORMAT = "16:9"

# Padding is expressed in pixels of a 24 px reference cell and scaled per format.
REFERENCE_CELL = 24
MAX_PADDING = REFERENCE_CELL // 2 - 1
DEFAULT_PADDING = 8

# Defaults you can override with --density, --shade-min and --shade-max.
DENSITY_DEFAULT = 0.55    # probability that a walker lights the cell it steps on (more = more dots)
SHADE_MIN_DEFAULT = 0.45  # brightness at a group's edge (0..1)
SHADE_MAX_DEFAULT = 1.0   # brightness at a group's center (0..1)

# Fixed shading details.
BRIGHTNESS_JITTER = 0.2   # random variation per dot (+/-)
MIN_BRIGHTNESS = 0.3      # floor so no dot disappears (lowered if shade-min is lower)

# Demo color: suggested when you press Enter in the menu. Never applied silently.
DEMO_COLOR = "#39FF14"

MAX_COLORS = 4

# Special color value that turns on rainbow mode.
RAINBOW = "rainbow"
HUE_SPAN_DEFAULT = 0.75   # fraction of the hue circle the gradient covers (1.0 = loops back)
HUE_JITTER = 0.03         # random hue variation per dot (fraction of the hue circle)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PRESETS_FILE = Path(__file__).resolve().parent / "presets.json"


def check_density(value: float) -> float:
    if not 0.05 <= value <= 1.0:
        raise argparse.ArgumentTypeError("density must be between 0.05 and 1.0")
    return value


def check_shading(shade_min: float, shade_max: float) -> None:
    if not (0.0 <= shade_min <= 1.0 and 0.0 <= shade_max <= 1.0):
        raise argparse.ArgumentTypeError("shading values must be between 0 and 1")
    if shade_min > shade_max:
        raise argparse.ArgumentTypeError("shade-min cannot be greater than shade-max")


def non_negative_int(value: str) -> int:
    """argparse type: accepts only integers >= 0."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"seed must be an integer, got {value!r}")
    if number < 0:
        raise argparse.ArgumentTypeError("seed must be zero or a positive integer")
    return number


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Accepts hex (#FF6A00 or FF6A00) or a color name (red, orange, teal...)."""
    c = color.strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", c):
        h = c.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if re.fullmatch(r"[A-Za-z]+", c):
        try:
            return tuple(ImageColor.getrgb(c.lower())[:3])
        except ValueError:
            pass
    raise ValueError(f"Invalid color: {color!r} (use hex like #FF6A00 or a name like red)")


def is_rainbow(color: str | None) -> bool:
    return color is not None and color.lower() == RAINBOW


def split_colors(color: str) -> list[str]:
    """'#FF0000, #00FF00' -> ['#FF0000', '#00FF00'] (1 to MAX_COLORS items)."""
    parts = [c.strip() for c in color.split(",") if c.strip()]
    if not 1 <= len(parts) <= MAX_COLORS:
        raise ValueError(f"Give between 1 and {MAX_COLORS} colors separated by commas")
    return parts


def parse_color(color: str) -> list[tuple[int, int, int]] | None:
    """Color string -> list of 1 to 4 RGB tuples, or None for rainbow mode."""
    if is_rainbow(color):
        return None
    return [hex_to_rgb(c) for c in split_colors(color)]


def normalize_color(color: str) -> str:
    """Canonical form: 'rainbow', or 1 to 4 uppercase hex colors joined by commas."""
    if is_rainbow(color):
        return RAINBOW
    return ",".join("#%02X%02X%02X" % hex_to_rgb(c) for c in split_colors(color))


def hsv_to_rgb(h: np.ndarray, s: float, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV -> RGB. h and v are arrays of the same shape, values 0..1.
    Returns float32 array (..., 3) with values 0..255."""
    i = np.floor(h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.float32)


def format_grid(format_name: str) -> tuple[int, int, int, int, int]:
    """Returns (width, height, cell, columns, rows) for a format."""
    width, height, cell = FORMATS[format_name]
    return width, height, cell, width // cell, height // cell


def padding_in_px(padding: int, cell: int) -> int:
    """Converts the reference padding to real pixels for the format's cell size."""
    return round(padding * cell / REFERENCE_CELL)


def walk_cells(rng: random.Random, rows: int, cols: int, density: float) -> np.ndarray:
    """Seeded walkers. Returns how many times each cell was lit."""
    hits = np.zeros((rows, cols), dtype=np.int32)
    walkers = rng.randint(3, 8)
    steps_per_walker = rng.randint(400, 1400)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    for _ in range(walkers):
        x = rng.randrange(cols)
        y = rng.randrange(rows)
        # Each walker has a bias direction, which forms elongated clusters
        bias = rng.choice(directions)
        for _ in range(steps_per_walker):
            if rng.random() < 0.35:
                dx, dy = bias
            else:
                dx, dy = rng.choice(directions)
            x = (x + dx) % cols
            y = (y + dy) % rows
            if rng.random() < density:
                hits[y, x] += 1
    return hits


def compute_brightness(lit: np.ndarray, nrng: np.random.Generator,
                       shade_min: float, shade_max: float) -> np.ndarray:
    """Brightness 0..1 per cell, water-like.
    Depth = fraction of lit neighbors (0..8). Group centers (8 neighbors) are
    at 100 %; edges dim. Random jitter makes some dots drop lower than others.
    """
    neighbors = np.zeros(lit.shape, dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbors += np.roll(np.roll(lit.astype(np.int32), dy, axis=0), dx, axis=1)

    depth = neighbors / 8.0
    brightness = shade_min + (shade_max - shade_min) * depth
    brightness += nrng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER, lit.shape)
    brightness = np.clip(brightness, min(MIN_BRIGHTNESS, shade_min), shade_max)
    return np.where(lit, brightness, 0.0).astype(np.float32)


def rainbow_defaults(seed: int) -> tuple[np.random.Generator, float, float]:
    """Seeded start hue and direction (degrees, 1 decimal so they fit in a file name),
    plus the generator positioned to draw the per-dot jitter afterwards."""
    hue_rng = np.random.default_rng([seed, 1])
    start = round(float(hue_rng.uniform(0.0, 360.0)), 1)
    angle = round(float(hue_rng.uniform(0.0, 360.0)), 1)
    return hue_rng, start, angle


def blend_colors(colors: list[tuple[int, int, int]], color_seed: int, rows: int, cols: int) -> np.ndarray:
    """Places each color at a random point and blends them with inverse-distance weights.
    Returns float32 (rows, cols, 3). Positions depend only on color_seed."""
    crng = np.random.default_rng([color_seed, 2])
    points = crng.uniform(0.0, 1.0, (len(colors), 2))  # (x, y) in 0..1
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    aspect = cols / rows
    weights = []
    for px, py in points:
        d2 = ((xx / cols - px) * aspect) ** 2 + (yy / rows - py) ** 2
        weights.append(1.0 / (d2 + 1e-4) ** 1.5)  # steeper than 1/d^2 so each color has a clear zone
    weights = np.stack(weights, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ np.array(colors, dtype=np.float32)


def compute_cells(seed: int, color: list[tuple[int, int, int]] | None, format_name: str,
                  density: float, shade_min: float, shade_max: float,
                  hue_start: float | None, hue_span: float, hue_angle: float | None,
                  color_seed: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Everything that does not depend on padding: (lit, brightness, cell_colors) per cell.
    cell_colors is float32 (rows, cols, 3) at full brightness."""
    width, height, _, cols, rows = format_grid(format_name)

    rng = random.Random(seed)
    hits = walk_cells(rng, rows, cols, density)
    lit = hits > 0
    brightness = compute_brightness(lit, np.random.default_rng(seed), shade_min, shade_max)

    # Color of each cell: one fixed color, a blend of random points, or a rainbow gradient.
    # Colors use their own generators so the dot layout never changes with them.
    if color is None:
        # Seeded start/angle are always drawn so overriding one does not change the others
        hue_rng, seeded_start, seeded_angle = rainbow_defaults(seed)
        jitter = hue_rng.uniform(-HUE_JITTER, HUE_JITTER, (rows, cols))
        start = (seeded_start if hue_start is None else hue_start) / 360.0
        angle = np.radians(seeded_angle if hue_angle is None else hue_angle)
        # Position along the gradient direction, normalized to 0..1 over the image
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
        proj = (xx / cols) * np.cos(angle) * (width / height) + (yy / rows) * np.sin(angle)
        t = (proj - proj.min()) / max(float(proj.max() - proj.min()), 1e-6)
        hue = (start + hue_span * t + jitter) % 1.0
        cell_colors = hsv_to_rgb(hue, 1.0, np.ones_like(hue))
    elif len(color) == 1:
        cell_colors = np.broadcast_to(np.array(color[0], dtype=np.float32), (rows, cols, 3))
    else:
        cell_colors = blend_colors(color, seed if color_seed is None else color_seed, rows, cols)
    return lit, brightness, cell_colors


def generate_wallpaper(seed: int, color: list[tuple[int, int, int]] | None, padding: int, format_name: str,
                       density: float = DENSITY_DEFAULT,
                       shade_min: float = SHADE_MIN_DEFAULT,
                       shade_max: float = SHADE_MAX_DEFAULT,
                       hue_start: float | None = None,
                       hue_span: float = HUE_SPAN_DEFAULT,
                       hue_angle: float | None = None,
                       color_seed: int | None = None) -> Image.Image:
    width, height, cell, _, _ = format_grid(format_name)
    pad = padding_in_px(padding, cell)
    side = cell - 2 * pad
    if side < 1:
        raise ValueError("Padding is too large for the cell size")

    lit, brightness, cell_colors = compute_cells(seed, color, format_name, density, shade_min, shade_max,
                                                 hue_start, hue_span, hue_angle, color_seed)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw each dot as a square of `side` px, centered in its cell, with its brightness
    for y, x in zip(*np.nonzero(lit)):
        y0 = y * cell + pad
        x0 = x * cell + pad
        canvas[y0:y0 + side, x0:x0 + side] = (cell_colors[y, x] * brightness[y, x]).astype(np.uint8)

    return Image.fromarray(canvas, mode="RGB")


def preview_wallpaper(seed: int, color: list[tuple[int, int, int]] | None, format_name: str,
                      density: float, shade_min: float, shade_max: float,
                      hue_start: float | None, hue_span: float, hue_angle: float | None,
                      color_seed: int | None) -> Image.Image:
    """Fast preview: one pixel per cell, no 8K canvas. Same dots and colors as the real image."""
    lit, brightness, cell_colors = compute_cells(seed, color, format_name, density, shade_min, shade_max,
                                                 hue_start, hue_span, hue_angle, color_seed)
    pixels = np.where(lit[..., None], cell_colors * brightness[..., None], 0.0)
    return Image.fromarray(pixels.astype(np.uint8), mode="RGB")


def terminal_preview(image: Image.Image) -> str:
    """Draws the image in the terminal with 24-bit color half-blocks.
    Each character shows two pixel rows: the top one as foreground, the bottom as background.
    The biggest size that fits the terminal (width and height) is used."""
    term = shutil.get_terminal_size((120, 40))
    width, height = image.size
    max_lines = max(term.lines - 4, 4)  # leave room for the prompt below
    cols = term.columns - 1

    # If the image would be too tall, shrink the columns until it fits
    if height / width * cols / 2 > max_lines:
        cols = int(max_lines * 2 * width / height)
    cols = max(cols, 1)

    rows = round(height / width * cols)
    rows += rows % 2  # needs an even number of rows (two per character)
    # NEAREST keeps the dots' real colors; BOX would average them with the black background
    small = image.resize((cols, rows), Image.NEAREST)
    pixels = small.load()

    lines = []
    for y in range(0, rows, 2):
        parts = []
        for x in range(cols):
            tr, tg, tb = pixels[x, y]
            br, bg, bb = pixels[x, y + 1]
            parts.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
        lines.append("".join(parts) + "\x1b[0m")
    return "\n".join(lines)


def output_name(seed: int, color: str, padding: int, format_name: str, density: float,
                shade_min: float, shade_max: float, hue_start: float | None, hue_span: float,
                hue_angle: float | None, color_seed: int | None) -> str:
    """File name holding every value needed to reproduce the image (no timestamp).
    Values that default to the seed (hue start/angle, color seed) are written out in full."""
    if is_rainbow(color):
        _, seeded_start, seeded_angle = rainbow_defaults(seed)
        start = seeded_start if hue_start is None else hue_start
        angle = seeded_angle if hue_angle is None else hue_angle
        color_part = f"rainbow-start{start:g}-span{hue_span:g}-angle{angle:g}"
    else:
        colors = split_colors(color)
        color_part = "hex" + "-".join(c.lstrip("#").upper() for c in colors)
        if len(colors) > 1:
            color_part += f"_cseed{seed if color_seed is None else color_seed}"
    aspect = format_name.replace(":", "-")
    return (f"pixels_wallpaper_8K_{aspect}_seed{seed}_{color_part}"
            f"_density{density:g}_min{shade_min:g}_max{shade_max:g}_padding{padding}.png")


@dataclass
class Settings:
    """Everything that defines a wallpaper. Used by the command line and the menu."""
    seed: int | None = None
    color: str | None = None
    format: str = DEFAULT_FORMAT
    padding: int = DEFAULT_PADDING
    density: float = DENSITY_DEFAULT
    shade_min: float = SHADE_MIN_DEFAULT
    shade_max: float = SHADE_MAX_DEFAULT
    hue_start: float | None = None
    hue_span: float = HUE_SPAN_DEFAULT
    hue_angle: float | None = None
    color_seed: int | None = None
    output: Path = OUTPUT_DIR

    def validate(self) -> None:
        """Raises ValueError with a readable message. Also normalizes the color."""
        if self.seed is None:
            raise ValueError("A seed is required (--seed N, or set it in the menu)")
        if self.color is None:
            raise ValueError(f"A color is required, e.g. {DEMO_COLOR}, up to {MAX_COLORS} separated "
                             f"by commas, or '{RAINBOW}'")
        self.color = normalize_color(self.color)
        if not 0 <= self.padding <= MAX_PADDING:
            raise ValueError(f"Padding must be between 0 and {MAX_PADDING}")
        if not 0.1 <= self.hue_span <= 1.0:
            raise ValueError("Hue span must be between 0.1 and 1.0")
        try:
            check_density(self.density)
            check_shading(self.shade_min, self.shade_max)
        except argparse.ArgumentTypeError as e:
            raise ValueError(str(e))

    def filename(self) -> str:
        return output_name(self.seed, self.color, self.padding, self.format, self.density,
                           self.shade_min, self.shade_max, self.hue_start, self.hue_span,
                           self.hue_angle, self.color_seed)

    def image(self) -> Image.Image:
        return generate_wallpaper(self.seed, parse_color(self.color), self.padding, self.format,
                                  self.density, self.shade_min, self.shade_max, self.hue_start,
                                  self.hue_span, self.hue_angle, self.color_seed)

    def preview(self) -> Image.Image:
        return preview_wallpaper(self.seed, parse_color(self.color), self.format, self.density,
                                 self.shade_min, self.shade_max, self.hue_start, self.hue_span,
                                 self.hue_angle, self.color_seed)

    def command(self) -> str:
        """Command line that reproduces this exact image. Seed-derived values are written out."""
        parts = ["python3 wallpaper_generator.py", f"--seed {self.seed}"]
        if is_rainbow(self.color):
            _, seeded_start, seeded_angle = rainbow_defaults(self.seed)
            start = seeded_start if self.hue_start is None else self.hue_start
            angle = seeded_angle if self.hue_angle is None else self.hue_angle
            parts += ["--rainbow", f"--hue-start {start:g}", f"--hue-span {self.hue_span:g}",
                      f"--hue-angle {angle:g}"]
        else:
            parts.append(f'--color "{self.color}"')
            if len(split_colors(self.color)) > 1:
                parts.append(f"--color-seed {self.seed if self.color_seed is None else self.color_seed}")
        parts += [f"--format {self.format}", f"--padding {self.padding}", f"--density {self.density:g}",
                  f"--shade-min {self.shade_min:g}", f"--shade-max {self.shade_max:g}"]
        if Path(self.output) != OUTPUT_DIR:
            parts.append(f'--output "{self.output}"')
        return " ".join(parts)

    def save_image(self) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / self.filename()
        self.image().save(path, optimize=True)
        return path

    def reset_style(self) -> None:
        """Back to defaults for everything except seed and color."""
        defaults = Settings()
        for f in fields(self):
            if f.name not in ("seed", "color"):
                setattr(self, f.name, getattr(defaults, f.name))


def load_presets() -> dict[str, dict]:
    try:
        data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        print(f"  Could not read {PRESETS_FILE.name}; ignoring it.")
        return {}


def save_preset(name: str, settings: Settings) -> None:
    presets = load_presets()
    data = asdict(settings)
    data["output"] = str(settings.output)
    presets[name] = data
    PRESETS_FILE.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def settings_from_preset(data: dict) -> Settings:
    known = {f.name for f in fields(Settings)}
    values = {k: v for k, v in data.items() if k in known}
    if "output" in values:
        values["output"] = Path(values["output"])
    return Settings(**values)


def ask(prompt: str) -> str:
    """input() that exits cleanly on Ctrl+C or Ctrl+D."""
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit("Cancelled.")


def ask_number(label: str, current, low: float, high: float, as_int: bool = False, allow_seed: bool = False):
    """Prompts for a number in [low, high]. Enter keeps the current value;
    'seed' (when allowed) goes back to deriving it from the seed. Raises ValueError."""
    shown = "from seed" if current is None else f"{current:g}"
    hint = ", 'seed' = from seed" if allow_seed else ""
    entry = ask(f"{label} [{shown}{hint}]: ").strip().lower()
    if not entry:
        return current
    if allow_seed and entry == "seed":
        return None
    value = int(entry) if as_int else float(entry)
    if not low <= value <= high:
        raise ValueError(f"must be between {low:g} and {high:g}")
    return value


def interactive_menu(st: Settings) -> None:
    """Menu: change what you want, then generate as many times as you like."""

    def color_details_txt() -> str:
        if st.color is None:
            return "set a color first"
        if is_rainbow(st.color):
            start = "from seed" if st.hue_start is None else f"{st.hue_start:g} deg"
            angle = "from seed" if st.hue_angle is None else f"{st.hue_angle:g} deg"
            return f"rainbow: start {start}, span {st.hue_span:g}, angle {angle}"
        if len(split_colors(st.color)) > 1:
            return f"color seed {'same as seed' if st.color_seed is None else st.color_seed}"
        return "nothing to set with a single color"

    def show_state() -> None:
        width, height, _, _, _ = format_grid(st.format)
        print("\n--- Current settings ---")
        print(f"  1) Seed         : {'not set' if st.seed is None else st.seed}")
        print(f"  2) Color(s)     : {'not set' if st.color is None else st.color}")
        print(f"  3) Color details: {color_details_txt()}")
        print(f"  4) Format       : {st.format}  ({width} x {height})")
        print(f"  5) Padding      : {st.padding}")
        print(f"  6) Density      : {st.density:g}")
        print(f"  7) Shading      : min {st.shade_min:g}  max {st.shade_max:g}")
        print(f"  8) Output folder: {st.output}")
        if st.seed is not None and st.color is not None:
            try:
                color = normalize_color(st.color)
                print("  File name       : " + output_name(st.seed, color, st.padding, st.format, st.density,
                                                           st.shade_min, st.shade_max, st.hue_start,
                                                           st.hue_span, st.hue_angle, st.color_seed))
            except ValueError:
                pass
        print("  p) Preview in terminal (saves nothing)    c) Show the command for these settings")
        print("  g) Generate and save (stay in the menu)   d) Reset everything except seed and color")
        print("  s) Save these settings as a preset        l) Load a preset")
        print("  q) Quit")

    while True:
        show_state()
        choice = ask("Choose (1-8/p/g/c/d/s/l/q): ").strip().lower()

        if choice == "1":
            entry = ask("New seed (non-negative integer, or 'random'): ").strip().lower()
            if entry == "random":
                st.seed = secrets.randbelow(1_000_000)
                print(f"  Random seed: {st.seed}")
            else:
                try:
                    st.seed = non_negative_int(entry)
                except argparse.ArgumentTypeError as e:
                    print(f"  {e}")
        elif choice == "2":
            entry = ask(f"New color: hex (#FF6A00) or name (red); up to {MAX_COLORS} separated by commas "
                        f"(#FF0000,blue); '{RAINBOW}' for rainbow; Enter = demo {DEMO_COLOR}: ").strip()
            try:
                st.color = normalize_color(entry or DEMO_COLOR)
                print(f"  Color: {st.color}")
            except ValueError as e:
                print(f"  {e}")
        elif choice == "3":
            if st.color is None:
                print("  Set a color first (option 2).")
            elif is_rainbow(st.color):
                try:
                    new_start = ask_number("Start hue in degrees, 0 to 360", st.hue_start, 0, 360, allow_seed=True)
                    new_span = ask_number("Span, share of the hue circle, 0.1 to 1.0 (1.0 loops back)",
                                          st.hue_span, 0.1, 1.0)
                    new_angle = ask_number("Direction in degrees (0 = left to right, 90 = top to bottom)",
                                           st.hue_angle, 0, 360, allow_seed=True)
                    st.hue_start, st.hue_span, st.hue_angle = new_start, new_span, new_angle
                except ValueError as e:
                    print(f"  Invalid value: {e}")
            elif len(split_colors(st.color)) > 1:
                try:
                    st.color_seed = ask_number("Color seed, non-negative integer", st.color_seed, 0, 2**32 - 1,
                                               as_int=True, allow_seed=True)
                except ValueError as e:
                    print(f"  Invalid value: {e}")
            else:
                print("  A single color has nothing to set. Give 2 to 4 colors or 'rainbow' in option 2.")
        elif choice == "4":
            names = list(FORMATS)
            print("  Available formats:")
            for letter, name in zip(string.ascii_lowercase, names):
                w, h, _, _, _ = format_grid(name)
                print(f"    {letter}) {name}  ({w} x {h})")
            entry = ask("Pick a letter or ratio (Enter = keep current): ").strip().lower()
            if entry:
                by_letter = dict(zip(string.ascii_lowercase, names))
                if entry in by_letter:
                    st.format = by_letter[entry]
                elif entry in FORMATS:
                    st.format = entry
                else:
                    print("  Invalid format.")
        elif choice == "5":
            try:
                st.padding = ask_number(f"Padding, 0 to {MAX_PADDING}", st.padding, 0, MAX_PADDING, as_int=True)
            except ValueError as e:
                print(f"  Invalid value: {e}")
        elif choice == "6":
            try:
                st.density = ask_number("Density 0.05 to 1.0, higher = more dots", st.density, 0.05, 1.0)
            except ValueError as e:
                print(f"  Invalid value: {e}")
        elif choice == "7":
            try:
                new_min = ask_number("Shading min, edge brightness 0 to 1", st.shade_min, 0, 1)
                new_max = ask_number("Shading max, center brightness 0 to 1", st.shade_max, 0, 1)
                check_shading(new_min, new_max)
                st.shade_min, st.shade_max = new_min, new_max
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"  Invalid value: {e}")
        elif choice == "8":
            entry = ask(f"Output folder [{st.output}]: ").strip()
            if entry:
                st.output = Path(entry).expanduser()
        elif choice in ("p", "g", "c"):
            try:
                st.validate()
            except ValueError as e:
                print(f"  {e}")
                continue
            if choice == "p":
                print(terminal_preview(st.preview()))
                ask("\n(Preview of the dot layout and colors. Press Enter to return to the menu)")
            elif choice == "c":
                print("\n" + st.command())
            else:
                path = st.save_image()
                print(f"\nSaved: {path}\nCommand: {st.command()}")
                print("Back in the menu: change anything and generate again.")
        elif choice == "d":
            st.reset_style()
            print("  Format, padding, density, shading, color details and output folder are back to defaults.")
        elif choice == "s":
            name = ask("Preset name: ").strip()
            if not name:
                print("  No name given, nothing saved.")
                continue
            save_preset(name, st)
            print(f"  Saved preset {name!r} in {PRESETS_FILE.name}")
        elif choice == "l":
            presets = load_presets()
            if not presets:
                print("  No presets saved yet. Use 's' to save one.")
                continue
            names = list(presets)
            for i, name in enumerate(names, start=1):
                print(f"    {i}) {name}")
            entry = ask("Pick a number or name (Enter = cancel): ").strip()
            if not entry:
                continue
            if entry.isdigit() and 1 <= int(entry) <= len(names):
                entry = names[int(entry) - 1]
            if entry not in presets:
                print("  No such preset.")
                continue
            try:
                loaded = settings_from_preset(presets[entry])
            except TypeError:
                print("  That preset is damaged.")
                continue
            for f in fields(Settings):
                setattr(st, f.name, getattr(loaded, f.name))
            print(f"  Loaded preset {entry!r}.")
        elif choice == "q":
            print("Bye.")
            return
        else:
            print("  Invalid option.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-random pixel art wallpaper, 8K height.")
    parser.add_argument("--seed", type=non_negative_int, default=None,
                        help="seed chosen by you; same seed and settings give the same image")
    parser.add_argument("--color", default=None,
                        help=f"dot color: hex or name; up to {MAX_COLORS} separated by commas to blend them, "
                             f"or '{RAINBOW}'; required (demo: {DEMO_COLOR})")
    parser.add_argument("--rainbow", action="store_true",
                        help="rainbow gradient across the image (same as --color rainbow)")
    parser.add_argument("--color-seed", type=non_negative_int, default=None,
                        help="seed for where each color sits when using 2-4 colors "
                             "(default: same as --seed)")
    parser.add_argument("--hue-start", type=float, default=None,
                        help="rainbow: starting hue in degrees 0-360 (default: from the seed)")
    parser.add_argument("--hue-span", type=float, default=HUE_SPAN_DEFAULT,
                        help=f"rainbow: how much of the hue circle the gradient covers, 0.1 to 1.0 "
                             f"(default {HUE_SPAN_DEFAULT}; 1.0 loops back to the first color)")
    parser.add_argument("--hue-angle", type=float, default=None,
                        help="rainbow: gradient direction in degrees (0 = left to right, "
                             "90 = top to bottom; default: from the seed)")
    parser.add_argument("--padding", type=int, default=DEFAULT_PADDING,
                        help=f"gap inside each cell (0 to {MAX_PADDING})")
    parser.add_argument("--format", default=DEFAULT_FORMAT, choices=list(FORMATS),
                        help="aspect ratio of the output (all at 8K height)")
    parser.add_argument("--density", type=float, default=DENSITY_DEFAULT,
                        help=f"how many dots (0.05 to 1.0, default {DENSITY_DEFAULT})")
    parser.add_argument("--shade-min", type=float, default=SHADE_MIN_DEFAULT,
                        help=f"brightness at group edges, 0 to 1 (default {SHADE_MIN_DEFAULT})")
    parser.add_argument("--shade-max", type=float, default=SHADE_MAX_DEFAULT,
                        help=f"brightness at group centers, 0 to 1 (default {SHADE_MAX_DEFAULT})")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output folder")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="menu to set everything, preview and generate")
    parser.add_argument("--preview", action="store_true",
                        help="draw the result in the terminal instead of saving a file")
    args = parser.parse_args()

    st = Settings(seed=args.seed, color=RAINBOW if args.rainbow else args.color, format=args.format,
                  padding=args.padding, density=args.density, shade_min=args.shade_min,
                  shade_max=args.shade_max, hue_start=args.hue_start, hue_span=args.hue_span,
                  hue_angle=args.hue_angle, color_seed=args.color_seed, output=args.output)

    if args.interactive:
        interactive_menu(st)
        return

    try:
        st.validate()
    except ValueError as e:
        parser.error(str(e))

    if args.preview:
        print(terminal_preview(st.preview()))
        return
    print(st.save_image())


if __name__ == "__main__":
    main()
