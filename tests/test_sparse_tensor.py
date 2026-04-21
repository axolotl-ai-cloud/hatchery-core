# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for sparse TensorData encoding in tinker_compat."""

from __future__ import annotations

import pytest

from hatchery.core.tinker_compat import TensorData, _reshape_tensor_data


def test_dense_1d_passthrough():
    td = TensorData(data=[1.0, 2.0, 3.0], shape=[3])
    assert _reshape_tensor_data(td) == [1.0, 2.0, 3.0]


def test_dense_2d_reshape():
    td = TensorData(data=[1, 2, 3, 4, 5, 6], shape=[3, 2])
    result = _reshape_tensor_data(td)
    assert result == [[1, 2], [3, 4], [5, 6]]


def test_sparse_1d_basic():
    """Sparse 1-D: 10 positions, only indices 2 and 7 are non-zero."""
    td = TensorData(
        shape=[10],
        sparse_col_indices=[2, 7],
        sparse_values=[1.0, 0.5],
    )
    result = _reshape_tensor_data(td)
    assert len(result) == 10
    assert result[2] == 1.0
    assert result[7] == 0.5
    assert result[0] == 0.0
    assert result[5] == 0.0


def test_sparse_1d_token_weights():
    """Typical use case: 32K sequence, last 500 tokens have weight=1.0."""
    seq_len = 32000
    start = 31500
    indices = list(range(start, seq_len))
    values = [1.0] * 500
    td = TensorData(
        shape=[seq_len],
        sparse_col_indices=indices,
        sparse_values=values,
    )
    result = _reshape_tensor_data(td)
    assert len(result) == seq_len
    assert all(v == 0.0 for v in result[:start])
    assert all(v == 1.0 for v in result[start:])


def test_sparse_1d_empty():
    """All zeros — no sparse entries."""
    td = TensorData(
        shape=[100],
        sparse_col_indices=[],
        sparse_values=[],
    )
    result = _reshape_tensor_data(td)
    assert len(result) == 100
    assert all(v == 0.0 for v in result)


def test_sparse_2d_csr():
    """Full 2-D CSR format."""
    # 3x4 matrix with entries at (0,1)=5, (1,0)=3, (1,2)=7, (2,3)=1
    td = TensorData(
        shape=[3, 4],
        sparse_crow_indices=[0, 1, 3, 4],
        sparse_col_indices=[1, 0, 2, 3],
        sparse_values=[5.0, 3.0, 7.0, 1.0],
    )
    result = _reshape_tensor_data(td)
    assert result == [
        [0.0, 5.0, 0.0, 0.0],
        [3.0, 0.0, 7.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_sparse_mismatched_lengths_raises():
    td = TensorData(
        shape=[10],
        sparse_col_indices=[1, 2, 3],
        sparse_values=[1.0, 2.0],  # length mismatch
    )
    with pytest.raises(ValueError, match="length"):
        _reshape_tensor_data(td)


def test_sparse_no_shape_raises():
    td = TensorData(
        sparse_col_indices=[1],
        sparse_values=[1.0],
    )
    with pytest.raises(ValueError, match="shape"):
        _reshape_tensor_data(td)


def test_sparse_out_of_bounds_ignored():
    """Indices beyond shape are silently ignored."""
    td = TensorData(
        shape=[5],
        sparse_col_indices=[1, 99],  # 99 is out of bounds
        sparse_values=[1.0, 2.0],
    )
    result = _reshape_tensor_data(td)
    assert len(result) == 5
    assert result[1] == 1.0
    # 99 is out of bounds — not placed.


def test_dense_takes_precedence_over_sparse():
    """If both data and sparse fields are set, sparse wins."""
    td = TensorData(
        data=[9, 9, 9],
        shape=[5],
        sparse_col_indices=[0],
        sparse_values=[1.0],
    )
    result = _reshape_tensor_data(td)
    # Sparse path runs when sparse_col_indices is set.
    assert len(result) == 5
    assert result[0] == 1.0
