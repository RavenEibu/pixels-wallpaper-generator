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
- The seed is always chosen by the user. It must be a non-negative integer.

Usage:
    python3 wallpaper_generator.py --seed 6
    python3 wallpaper_generator.py --seed 6 --format 21:9 --color "#FF6A00"
    python3 wallpaper_generator.py --seed 6 --rainbow
    python3 wallpaper_generator.py -i        # interactive menu
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

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


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", hex_color):
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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


def generate_wallpaper(seed: int, color: list[tuple[int, int, int]] | None, padding: int, format_name: str,
                       density: float = DENSITY_DEFAULT,
                       shade_min: float = SHADE_MIN_DEFAULT,
                       shade_max: float = SHADE_MAX_DEFAULT,
                       hue_start: float | None = None,
                       hue_span: float = HUE_SPAN_DEFAULT,
                       hue_angle: float | None = None,
                       color_seed: int | None = None) -> Image.Image:
    width, height, cell, cols, rows = format_grid(format_name)
    pad = padding_in_px(padding, cell)
    side = cell - 2 * pad
    if side < 1:
        raise ValueError("Padding is too large for the cell size")

    rng = random.Random(seed)
    hits = walk_cells(rng, rows, cols, density)
    lit = hits > 0
    brightness = compute_brightness(lit, np.random.default_rng(seed), shade_min, shade_max)

    # Color of each cell at full brightness: one fixed color, or a rainbow gradient.
    # Rainbow uses its own generator so the dot layout stays the same as without it.
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

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw each dot as a square of `side` px, centered in its cell, with its brightness
    for y, x in zip(*np.nonzero(lit)):
        y0 = y * cell + pad
        x0 = x * cell + pad
        canvas[y0:y0 + side, x0:x0 + side] = (cell_colors[y, x] * brightness[y, x]).astype(np.uint8)

    return Image.fromarray(canvas, mode="RGB")


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


def ask_number(label: str, current, low: float, high: float, as_int: bool = False, allow_seed: bool = False):
    """Prompts for a number in [low, high]. Enter keeps the current value;
    'seed' (when allowed) goes back to deriving it from the seed. Raises ValueError."""
    shown = "from seed" if current is None else f"{current:g}"
    hint = ", 'seed' = from seed" if allow_seed else ""
    entry = input(f"{label} [{shown}{hint}]: ").strip().lower()
    if not entry:
        return current
    if allow_seed and entry == "seed":
        return None
    value = int(entry) if as_int else float(entry)
    if not low <= value <= high:
        raise ValueError(f"must be between {low:g} and {high:g}")
    return value


def interactive_menu(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Menu: change what you want, then confirm with 'g' to generate."""
    seed = args.seed
    color = args.color
    padding = args.padding
    format_name = args.format
    density = args.density
    shade_min = args.shade_min
    shade_max = args.shade_max
    hue_start = args.hue_start
    hue_span = args.hue_span
    hue_angle = args.hue_angle
    color_seed = args.color_seed
    output = args.output

    def color_details_txt() -> str:
        if color is None:
            return "set a color first"
        if is_rainbow(color):
            start = "from seed" if hue_start is None else f"{hue_start:g} deg"
            angle = "from seed" if hue_angle is None else f"{hue_angle:g} deg"
            return f"rainbow: start {start}, span {hue_span:g}, angle {angle}"
        if len(split_colors(color)) > 1:
            return f"color seed {'same as seed' if color_seed is None else color_seed}"
        return "nothing to set with a single color"

    def show_state() -> None:
        width, height, _, _, _ = format_grid(format_name)
        print("\n--- Current settings ---")
        print(f"  1) Seed         : {'not set' if seed is None else seed}")
        print(f"  2) Color(s)     : {'not set' if color is None else color}")
        print(f"  3) Color details: {color_details_txt()}")
        print(f"  4) Format       : {format_name}  ({width} x {height})")
        print(f"  5) Padding      : {padding}")
        print(f"  6) Density      : {density:g}")
        print(f"  7) Shading      : min {shade_min:g}  max {shade_max:g}")
        print(f"  8) Output folder: {output}")
        if seed is not None and color is not None:
            print("  File name       : " + output_name(seed, color, padding, format_name, density,
                                                       shade_min, shade_max, hue_start, hue_span,
                                                       hue_angle, color_seed))
        print("  p) Preview in terminal (saves nothing)")
        print("  g) Generate with these settings")
        print("  q) Quit without generating")

    while True:
        show_state()
        choice = input("Choose (1-8/p/g/q): ").strip().lower()

        if choice == "1":
            entry = input("New seed (non-negative integer): ").strip()
            try:
                seed = non_negative_int(entry)
            except argparse.ArgumentTypeError as e:
                print(f"  {e}")
        elif choice == "2":
            entry = input(f"New hex color (e.g. #FF6A00; up to {MAX_COLORS} separated by commas, "
                          f"e.g. #FF0000,#0000FF; '{RAINBOW}' for rainbow; "
                          f"Enter = demo {DEMO_COLOR}): ").strip()
            if not entry:
                entry = DEMO_COLOR
            try:
                parse_color(entry)
                color = RAINBOW if is_rainbow(entry) else ",".join(split_colors(entry))
            except ValueError as e:
                print(f"  {e}")
        elif choice == "3":
            if color is None:
                print("  Set a color first (option 2).")
            elif is_rainbow(color):
                try:
                    new_start = ask_number("Start hue in degrees, 0 to 360", hue_start, 0, 360, allow_seed=True)
                    new_span = ask_number("Span, share of the hue circle, 0.1 to 1.0 (1.0 loops back)",
                                          hue_span, 0.1, 1.0)
                    new_angle = ask_number("Direction in degrees (0 = left to right, 90 = top to bottom)",
                                           hue_angle, 0, 360, allow_seed=True)
                    hue_start, hue_span, hue_angle = new_start, new_span, new_angle
                except ValueError as e:
                    print(f"  Invalid value: {e}")
            elif len(split_colors(color)) > 1:
                try:
                    color_seed = ask_number("Color seed, non-negative integer", color_seed, 0, 2**32 - 1,
                                            as_int=True, allow_seed=True)
                except ValueError as e:
                    print(f"  Invalid value: {e}")
            else:
                print("  A single color has nothing to set. Give 2 to 4 colors or 'rainbow' in option 2.")
        elif choice == "4":
            print("  Available formats:")
            names = list(FORMATS)
            for i, name in enumerate(names, start=1):
                w, h, _, _, _ = format_grid(name)
                print(f"    {i}) {name}  ({w} x {h})")
            entry = input("Pick a number or name (empty = keep current): ").strip()
            if entry:
                if entry.isdigit() and 1 <= int(entry) <= len(names):
                    format_name = names[int(entry) - 1]
                elif entry in FORMATS:
                    format_name = entry
                else:
                    print("  Invalid format.")
        elif choice == "5":
            try:
                padding = ask_number(f"Padding, 0 to {MAX_PADDING}", padding, 0, MAX_PADDING, as_int=True)
            except ValueError as e:
                print(f"  Invalid value: {e}")
        elif choice == "6":
            try:
                density = ask_number("Density 0.05 to 1.0, higher = more dots", density, 0.05, 1.0)
            except ValueError as e:
                print(f"  Invalid value: {e}")
        elif choice == "7":
            try:
                new_min = ask_number("Shading min, edge brightness 0 to 1", shade_min, 0, 1)
                new_max = ask_number("Shading max, center brightness 0 to 1", shade_max, 0, 1)
                check_shading(new_min, new_max)
                shade_min, shade_max = new_min, new_max
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"  Invalid value: {e}")
        elif choice == "8":
            entry = input(f"Output folder [{output}]: ").strip()
            if entry:
                output = Path(entry).expanduser()
        elif choice == "p":
            if seed is None or color is None:
                print("  Set a seed (option 1) and a color (option 2) first.")
                continue
            image = generate_wallpaper(seed, parse_color(color), padding, format_name,
                                       density, shade_min, shade_max,
                                       hue_start, hue_span, hue_angle, color_seed)
            print(terminal_preview(image))
            input("\n(Preview above. Press Enter to return to the menu)")
        elif choice == "g":
            if seed is None:
                print("  Set a seed first (option 1).")
                continue
            if color is None:
                print("  Set a color first (option 2).")
                continue
            args.seed, args.color, args.padding, args.format = seed, color, padding, format_name
            args.density, args.shade_min, args.shade_max = density, shade_min, shade_max
            args.hue_start, args.hue_span, args.hue_angle = hue_start, hue_span, hue_angle
            args.color_seed, args.output = color_seed, output
            return args
        elif choice == "q":
            raise SystemExit("Cancelled, nothing was generated.")
        else:
            print("  Invalid option.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo-random pixel art wallpaper, 8K height.")
    parser.add_argument("--seed", type=non_negative_int, default=None,
                        help="seed chosen by you; same seed and settings give the same image")
    parser.add_argument("--color", default=None,
                        help=f"dot color in hex, up to {MAX_COLORS} separated by commas to blend them, "
                             f"or '{RAINBOW}'; required (demo: {DEMO_COLOR})")
    parser.add_argument("--color-seed", type=non_negative_int, default=None,
                        help="seed for where each color sits when using 2-4 colors "
                             "(default: same as --seed)")
    parser.add_argument("--rainbow", action="store_true",
                        help="rainbow gradient across the width (same as --color rainbow)")
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
    parser.add_argument("--hue-start", type=float, default=None,
                        help="rainbow: starting hue in degrees 0-360 (default: from the seed)")
    parser.add_argument("--hue-span", type=float, default=HUE_SPAN_DEFAULT,
                        help=f"rainbow: how much of the hue circle the gradient covers, 0.1 to 1.0 "
                             f"(default {HUE_SPAN_DEFAULT}; 1.0 loops back to the first color)")
    parser.add_argument("--hue-angle", type=float, default=None,
                        help="rainbow: gradient direction in degrees (0 = left to right, "
                             "90 = top to bottom; default: from the seed)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="output folder")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="menu to set seed, color, padding and format before generating")
    parser.add_argument("--preview", action="store_true",
                        help="draw the result in the terminal instead of saving a file")
    args = parser.parse_args()

    if args.interactive:
        args = interactive_menu(args, parser)

    if args.rainbow:
        args.color = RAINBOW
    if args.seed is None:
        parser.error("--seed is required (or use -i)")
    if args.color is None:
        parser.error(f"--color is required, e.g. --color {DEMO_COLOR} (or use -i)")

    try:
        color = parse_color(args.color)
    except ValueError as e:
        parser.error(str(e))
    if not 0 <= args.padding <= MAX_PADDING:
        parser.error(f"--padding must be between 0 and {MAX_PADDING}")
    try:
        check_density(args.density)
        check_shading(args.shade_min, args.shade_max)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    if not 0.1 <= args.hue_span <= 1.0:
        parser.error("--hue-span must be between 0.1 and 1.0")

    settings = dict(density=args.density, shade_min=args.shade_min, shade_max=args.shade_max,
                    hue_start=args.hue_start, hue_span=args.hue_span, hue_angle=args.hue_angle,
                    color_seed=args.color_seed)

    if args.preview:
        print(terminal_preview(generate_wallpaper(args.seed, color, args.padding, args.format, **settings)))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / output_name(args.seed, args.color, args.padding, args.format, args.density,
                                     args.shade_min, args.shade_max, args.hue_start, args.hue_span,
                                     args.hue_angle, args.color_seed)

    image = generate_wallpaper(args.seed, color, args.padding, args.format, **settings)
    image.save(path, optimize=True)
    print(path)


if __name__ == "__main__":
    main()
