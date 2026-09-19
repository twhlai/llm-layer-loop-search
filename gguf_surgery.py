#!/usr/bin/env python3
"""
GGUF Layer Surgery

Reads a GGUF model file and a layer path, and writes a new GGUF with the
specified layer structure.

A negative layer number indicates that the corresponding positive layer
is included with its feed-forward network down-projection weights set to 0.  
Or, more simply, -14 means "include layer 14's attention sublayer, but
don't use layer 14's feed-forward network".

Tensor naming convention: blk.{layer_idx}.{tensor_name}
Non-block tensors (token_embd, output_norm, output) are copied as-is.
"""

import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType


BLK_PATTERN = re.compile(r'^blk\.(\d+)\.(.+)$')


def get_field_value(reader: GGUFReader, key: str):
    """Extract a scalar value from a reader field."""
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def build_gguf_from_path(input_path: str, output_path: str,
                         layer_path: list[int], verbose: bool = False):
    """
    Create a new GGUF where the forward pass follows the given layer path.
    """
    reader = GGUFReader(input_path, 'r')

    arch = get_field_value(reader, gguf.Keys.General.ARCHITECTURE)
    if arch is None:
        raise ValueError("Could not read architecture from GGUF")

    block_count_key = f'{arch}.block_count'
    orig_block_count = get_field_value(reader, block_count_key)
    if orig_block_count is None:
        raise ValueError(f"Could not read {block_count_key} from GGUF")

    # Validate all layer indices
    for idx in layer_path:
        if idx <= -orig_block_count or idx >= orig_block_count:
            raise ValueError(
                f"Layer {idx} out of range (model has {orig_block_count} layers, 0..{orig_block_count-1})"
            )

    new_block_count = len(layer_path)

    if verbose:
        print(f"Architecture: {arch}")
        print(f"Original layers: {orig_block_count}")
        print(f"New layer count: {new_block_count}")
        print(f"Layer path: {layer_path}")

    # layer_map: new_position -> original_layer_index
    layer_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(layer_path)}

    # Create writer
    writer = GGUFWriter(output_path, arch=arch, endianess=reader.endianess)

    alignment = get_field_value(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    # Copy metadata, override block count
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        if field.name == block_count_key:
            writer.add_key_value(field.name, new_block_count, val_type)
        else:
            val = field.contents()
            if val is not None:
                writer.add_key_value(field.name, val, val_type, sub_type=sub_type)

    # Organize tensors by layer
    non_block_tensors = []
    block_tensors = {}

    for tensor in reader.tensors:
        match = BLK_PATTERN.match(tensor.name)
        if match:
            layer_idx = int(match.group(1))
            suffix = match.group(2)
            if layer_idx not in block_tensors:
                block_tensors[layer_idx] = []
            block_tensors[layer_idx].append((suffix, tensor))
        else:
            non_block_tensors.append(tensor)

    pre_block = [t for t in non_block_tensors if 'output' not in t.name]
    post_block = [t for t in non_block_tensors if 'output' in t.name]

    # Add tensor infos and build write order
    total_bytes = 0
    block_write_order = []

    for tensor in pre_block:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        total_bytes += tensor.n_bytes

    for new_idx in range(new_block_count):
        orig_idx = layer_map[new_idx]
        zero_ffn = False
        if orig_idx < 0:
            orig_idx = -orig_idx
            zero_ffn = True
        if orig_idx not in block_tensors:
            raise ValueError(f"No tensors found for original layer {orig_idx}")
        for suffix, tensor in block_tensors[orig_idx]:
            new_name = f"blk.{new_idx}.{suffix}"
            writer.add_tensor_info(new_name, tensor.data.shape, tensor.data.dtype,
                                   tensor.data.nbytes, tensor.tensor_type)
            total_bytes += tensor.n_bytes
            if suffix == "ffn_down.weight" and zero_ffn:
                new_tensor = tensor._replace(
                    data=np.zeros_like(tensor.data)
                )
                block_write_order.append(new_tensor)
            else:
                block_write_order.append(tensor)

    for tensor in post_block:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        total_bytes += tensor.n_bytes

    # Write
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing GGUF", total=total_bytes, unit="B", unit_scale=True)

    for tensor in pre_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    for tensor in block_write_order:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    for tensor in post_block:
        writer.write_tensor_data(tensor.data)
        bar.update(tensor.n_bytes)

    bar.close()
    writer.close()

    if verbose:
        out_size = Path(output_path).stat().st_size / (1024**3)
        print(f"Done. Output: {out_size:.2f} GiB")
