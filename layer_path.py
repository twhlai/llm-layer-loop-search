#!/usr/bin/env python3
"""
Explicit Layer Path GGUF Surgery

You provide the exact sequence of layers the model should execute.

Examples:
    # Normal 40-layer model (identity, for testing)
    python layer_path.py model.gguf out.gguf -p 0..39

    # Duplicate layers 13-16 once
    python layer_path.py model.gguf out.gguf -p 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,13,14,15,16,17,18..39

    # Repeat layer 13 four times
    python layer_path.py model.gguf out.gguf -p 0,1..12,13,13,13,13,14,15..39

    # Repeat layer 10's and 20's attention sublayers but not feed-forward networks
    python layer_path.py model.gguf out.gguf -p 0,1..9,-10,10..19,-20,20..39

    # Shorthand: use .. to fill in sequential ranges
    python layer_path.py model.gguf out.gguf -p 0..16,13,14,15,16,13,14,15,16,17..39

Usage:
    python layer_path.py input.gguf output.gguf -p "0..16,13,14,15,16,17..39" -v
"""

import argparse
import re
from gguf_surgery import build_gguf_from_path


BLK_PATTERN = re.compile(r'^blk\.(\d+)\.(.+)$')


def get_field_value(reader, key):
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def parse_layer_list(path_str: str) -> list[int]:
    """
    Parse a layer list string into a list of layer indices.

    Supports:
        - Individual numbers: 0,1,2,13,13,14
        - Ranges with ..: 0..16 expands to 0,1,2,...,16 (inclusive)
        - Mixed: 0..12,13,13,13,14..39

    A negative number indicates layer is repeated with ffn down projection
    weights set to 0.

    Whitespace is ignored.
    """
    path_str = path_str.replace(' ', '')
    layers = []

    for part in path_str.split(','):
        part = part.strip()
        if not part:
            continue

        if '..' in part:
            # Range: start..end (inclusive)
            pieces = part.split('..')
            if len(pieces) != 2:
                raise ValueError(f"Invalid range: '{part}'. Use 'start..end'")
            start = int(pieces[0])
            end = int(pieces[1])
            if start > end:
                raise ValueError(f"Invalid range: {start}..{end} (start > end)")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))

    return layers


def main():
    parser = argparse.ArgumentParser(
        description="Build GGUF with explicit layer execution path",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Duplicate layers 13-16 once (RYS style)
  %(prog)s model.gguf out.gguf -p "0..16,13,14,15,16,17..39"

  # Repeat just layer 13 four times
  %(prog)s model.gguf out.gguf -p "0..12,13,13,13,13,14..39"

  # Repeat layers 10's and 20's attention sublayers but not feed-forward networks
  %(prog)s model.gguf out.gguf -p 0,1..9,-10,10..19,-20,20..39

  # Skip layer 5 entirely
  %(prog)s model.gguf out.gguf -p "0..4,6..39"
        """
    )
    parser.add_argument("input", help="Input GGUF file")
    parser.add_argument("output", help="Output GGUF file")
    parser.add_argument("-p", "--path", required=True,
                        help="Layer execution path (e.g. '0..16,13,14,15,16,17..39')")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    layer_path = parse_layer_list(args.path)
    print(f"Model: {args.input}")
    print(f"Output: {args.output}")
    print(f"Layer path ({len(layer_path)} layers): {layer_path}")

    build_gguf_from_path(args.input, args.output, layer_path, args.verbose)


if __name__ == "__main__":
    main()
