#!/usr/bin/env python3
"""Pseudo-random pixel art wallpaper generator (8K height). MVP.

- Output height is 8K (4320 px); width follows the chosen aspect ratio.
- Every format uses 48 px cells and 90 rows, so pixels scale cleanly.
- Each dot is a square smaller than its cell; the gap between dots is the padding.
- Pseudo-random: seeded walkers light up cells, forming organic clusters.
  Same seed + same format + same color + same padding = same image.
- Shading: cell brightness depends on how many lit neighbors it has, so
  group centers are brightest and edges dim, with some random variation.
- The seed is always chosen by the user. It must be a non-negative integer.

Usage:
    python3 wallpaper_generator.py --seed 6
    python3 wallpaper_generator.py --seed 6 --format 21:9 --color "#FF6A00"
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


def generate_wallpaper(seed: int, color: tuple[int, int, int], padding: int, format_name: str,
                       density: float = DENSITY_DEFAULT,
                       shade_min: float = SHADE_MIN_DEFAULT,
                       shade_max: float = SHADE_MAX_DEFAULT) -> Image.Image:
    width, height, cell, cols, rows = format_grid(format_name)
    pad = padding_in_px(padding, cell)
    side = cell - 2 * pad
    if side < 1:
        raise ValueError("Padding is too large for the cell size")

    rng = random.Random(seed)
    hits = walk_cells(rng, rows, cols, density)
    lit = hits > 0
    brightness = compute_brightness(lit, np.random.default_rng(seed), shade_min, shade_max)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    base = np.array(color, dtype=np.float32)
    # Draw each dot as a square of `side` px, centerd in its cell, with its brightness
    for y, x in zip(*np.nonzero(lit)):
        y0 = y * cell + pad
        x0 = x * cell + pad
        canvas[y0:y0 + side, x0:x0 + side] = (base * brightness[y, x]).astype(np.uint8)

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


def interactive_menu(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Menu: change what you want, then confirm with 'g' to generate."""
    seed = args.seed
    color = args.color
    padding = args.padding
    format_name = args.format
    density = args.density
    shade_min = args.shade_min
    shade_max = args.shade_max

    def show_state() -> None:
        seed_txt = "not set" if seed is None else str(seed)
        color_txt = "not set" if color is None else color
        width, height, _, _, _ = format_grid(format_name)
        print("\n--- Current settings ---")
        print(f"  1) Seed       : {seed_txt}")
        print(f"  2) Color      : {color_txt}")
        print(f"  3) Padding    : {padding}")
        print(f"  4) Format     : {format_name}  ({width} x {height})")
        print(f"  5) Shading    : min {shade_min}  max {shade_max}")
        print(f"  6) Density    : {density}")
        print("  p) Preview in terminal (saves nothing)")
        print("  g) Generate with these settings")
        print("  q) Quit without generating")

    while True:
        show_state()
        choice = input("Choose (1-6/p/g/q): ").strip().lower()

        if choice == "5":
            entry_min = input(f"Shading min, edge brightness 0 to 1 [{shade_min}]: ").strip()
            entry_max = input(f"Shading max, center brightness 0 to 1 [{shade_max}]: ").strip()
            try:
                new_min = float(entry_min) if entry_min else shade_min
                new_max = float(entry_max) if entry_max else shade_max
                check_shading(new_min, new_max)
                shade_min, shade_max = new_min, new_max
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"  {e}")
            continue
        if choice == "6":
            entry = input(f"Density 0.05 to 1.0 (higher = more dots) [{density}]: ").strip()
            try:
                density = check_density(float(entry)) if entry else density
            except (ValueError, argparse.ArgumentTypeError) as e:
                print(f"  {e}")
            continue

        if choice == "1":
            entry = input("New seed (non-negative integer): ").strip()
            try:
                seed = non_negative_int(entry)
            except argparse.ArgumentTypeError as e:
                print(f"  {e}")
        elif choice == "2":
            entry = input(f"New hex color (e.g. #FF6A00, Enter = demo {DEMO_COLOR}): ").strip()
            if not entry:
                entry = DEMO_COLOR
            try:
                hex_to_rgb(entry)
                color = entry
            except ValueError as e:
                print(f"  {e}")
        elif choice == "3":
            entry = input(f"New padding (0 to {MAX_PADDING}): ").strip()
            try:
                value = int(entry)
                if 0 <= value <= MAX_PADDING:
                    padding = value
                else:
                    print(f"  Padding must be between 0 and {MAX_PADDING}.")
            except ValueError:
                print("  Padding must be an integer.")
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
        elif choice == "p":
            if seed is None or color is None:
                print("  Set a seed (option 1) and a color (option 2) first.")
                continue
            image = generate_wallpaper(seed, hex_to_rgb(color), padding, format_name,
                                       density, shade_min, shade_max)
            print(terminal_preview(image))
            input("\n(Preview above. Press Enter to return to the menu)")
        elif choice == "g":
            if seed is None:
                print("  Set a seed first (option 1).")
                continue
            if color is None:
                print("  Set a color first (option 2).")
                continue
            args.seed = seed
            args.color = color
            args.padding = padding
            args.format = format_name
            args.density = density
            args.shade_min = shade_min
            args.shade_max = shade_max
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
                        help=f"dot color in hex, required (demo: {DEMO_COLOR})")
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
                        help="menu to set seed, color, padding and format before generating")
    parser.add_argument("--preview", action="store_true",
                        help="draw the result in the terminal instead of saving a file")
    args = parser.parse_args()

    if args.interactive:
        args = interactive_menu(args, parser)

    if args.seed is None:
        parser.error("--seed is required (or use -i)")
    if args.color is None:
        parser.error(f"--color is required, e.g. --color {DEMO_COLOR} (or use -i)")

    try:
        color = hex_to_rgb(args.color)
    except ValueError as e:
        parser.error(str(e))
    if not 0 <= args.padding <= MAX_PADDING:
        parser.error(f"--padding must be between 0 and {MAX_PADDING}")
    try:
        check_density(args.density)
        check_shading(args.shade_min, args.shade_max)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    settings = dict(density=args.density, shade_min=args.shade_min, shade_max=args.shade_max)

    if args.preview:
        print(terminal_preview(generate_wallpaper(args.seed, color, args.padding, args.format, **settings)))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    width, height, _, _, _ = format_grid(args.format)
    # Filename holds every value needed to reproduce the image (no timestamp)
    hex_digits = args.color.lstrip("#").upper()
    aspect = args.format.replace(":", "-")
    name = (f"pixels_wallpaper_8K_{aspect}_seed{args.seed}_hex{hex_digits}"
            f"_density{args.density}_min{args.shade_min}_max{args.shade_max}"
            f"_padding{args.padding}.png")
    path = args.output / name

    image = generate_wallpaper(args.seed, color, args.padding, args.format, **settings)
    image.save(path, optimize=True)
    print(path)


if __name__ == "__main__":
    main()
