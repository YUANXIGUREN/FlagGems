# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import contextlib
import importlib
import multiprocessing
from pathlib import Path

import pytest
import torch
import triton

import flag_gems
from flag_gems.utils import get_device_properties
from flag_gems.utils.codegen_config_utils import CodeGenConfig as CommonCodeGenConfig
from flag_gems.utils.pointwise_dynamic import (
    pointwise_dynamic as common_pointwise_dynamic,
)

if flag_gems.vendor_name == "cambricon":
    from flag_gems.runtime.backend._cambricon.utils.pointwise_dynamic import (
        CodeGenConfig,
        ComplexMode,
        FunctionSchema,
        pointwise_dynamic,
    )
else:
    from flag_gems.utils.pointwise_dynamic import (
        CodeGenConfig,
        ComplexMode,
        FunctionSchema,
        pointwise_dynamic,
    )

from flag_gems.utils.tensor_wrapper import StridedBuffer

MAX_GRID_SIZES = (65535, 65535, 65535)
MAX_GRID_SIZE_X = MAX_GRID_SIZES[0]

USE_BLOCK_POINTER = [True, False]
triton_version_less_than3 = int(triton.__version__[0]) < 3

if flag_gems.vendor_name == "kunlunxin":
    pytestmark = pytest.mark.skip("Issue #2836: not working")


@pytest.mark.parametrize(
    "num_tiles,max_grid,want",
    [(1, 3, (1, 1)), (4, 3, (2, 2)), (6, 3, (3, 2)), (7, 3, (3, 3))],
)
def test_balanced_grid_partition(num_tiles, max_grid, want):
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")
    assert helper(num_tiles, max_grid) == want


def test_balanced_grid_partition_rejects_nonpositive_tiles():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")
    with pytest.raises(ValueError, match="num_tiles must be positive"):
        helper(0, 3)


def test_balanced_grid_partition_rejects_nonpositive_max_grid():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    helper = getattr(module, "_balanced_grid_partition")
    with pytest.raises(ValueError, match="max_grid_size must be positive"):
        helper(4, 0)


def test_codegen_config_balance_grid_defaults_off():
    config = CommonCodeGenConfig(4, (3, 1, 1), 4, True, False)
    assert config.balance_grid is False


def test_codegen_config_accepts_balance_grid_opt_in():
    config = CommonCodeGenConfig(4, (3, 1, 1), 4, True, False, balance_grid=True)
    assert config.balance_grid is True


def test_balanced_grid_partition_covers_each_tile_once():
    module = importlib.import_module("flag_gems.utils.pointwise_dynamic")
    num_ctas, tiles_per_cta = getattr(module, "_balanced_grid_partition")(4, 3)
    tile_ids = [
        pid + round_id * num_ctas
        for round_id in range(tiles_per_cta)
        for pid in range(num_ctas)
        if pid + round_id * num_ctas < 4
    ]
    assert sorted(tile_ids) == [0, 1, 2, 3]
    assert len(tile_ids) == len(set(tile_ids))


@pytest.mark.parametrize("prefer_1d_tile", [False, True])
def test_balanced_grid_generated_wrapper_source_and_runtime_partition(
    monkeypatch, prefer_1d_tile
):
    @triton.jit
    def copy_scalar(x):
        return x

    default_config = CommonCodeGenConfig(4, (3, 1, 1), 4, False, prefer_1d_tile)
    balanced_config = CommonCodeGenConfig(
        4, (3, 1, 1), 4, False, prefer_1d_tile, balance_grid=True
    )
    default_fn = common_pointwise_dynamic(
        copy_scalar,
        num_inputs=1,
        promotion_methods=[(0, "DEFAULT")],
        config=default_config,
    )
    balanced_fn = common_pointwise_dynamic(
        copy_scalar,
        num_inputs=1,
        promotion_methods=[(0, "DEFAULT")],
        config=balanced_config,
    )

    default_info = default_fn.get_kernel_info(1)
    balanced_info = balanced_fn.get_kernel_info(1)
    default_source = Path(default_info.file_path).read_text()
    balanced_source = Path(balanced_info.file_path).read_text()

    helper_import = (
        "from flag_gems.utils.pointwise_dynamic import _balanced_grid_partition"
    )
    assert helper_import not in default_source
    assert "num_ctas = min(3, num_tiles)" in default_source
    assert "tiles_per_cta = triton.cdiv(num_tiles, num_ctas)" in default_source
    assert helper_import in balanced_source
    assert "num_ctas, tiles_per_cta = _balanced_grid_partition(num_tiles, 3)" in (
        balanced_source
    )

    observed_partitions = []

    class KernelLaunchSpy:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                observed_partitions.append((grid[0], kwargs["tiles_per_cta"]))

            return launch

    class NoopDeviceContext:
        @staticmethod
        def device(index):
            return contextlib.nullcontext()

    for fn, info in ((default_fn, default_info), (balanced_fn, balanced_info)):
        wrapper = fn.instantiate(1)
        monkeypatch.setitem(wrapper.__globals__, info.kernel_name, KernelLaunchSpy())
        monkeypatch.setitem(
            wrapper.__globals__, "heuristics_for_tile_size", lambda *args: (4,)
        )
        monkeypatch.setitem(
            wrapper.__globals__, "heuristics_for_num_warps", lambda tile_size: 1
        )
        monkeypatch.setitem(wrapper.__globals__, "torch_device_fn", NoopDeviceContext)
        source = torch.empty(16)
        destination = torch.empty_like(source)
        assert wrapper(source, out0=destination) is destination

    assert observed_partitions == [(3, 2), (2, 2)]


def test_balanced_grid_cache_isolation():
    @triton.jit
    def copy_scalar(x):
        return x

    default_config = CommonCodeGenConfig(4, (3, 1, 1), 4, False, True)
    same_default_config = CommonCodeGenConfig(4, (3, 1, 1), 4, False, True)
    balanced_config = CommonCodeGenConfig(
        4, (3, 1, 1), 4, False, True, balance_grid=True
    )

    def make_function(config):
        return common_pointwise_dynamic(
            copy_scalar,
            num_inputs=1,
            promotion_methods=[(0, "DEFAULT")],
            config=config,
        )

    default_fn = make_function(default_config)
    same_default_fn = make_function(same_default_config)
    balanced_fn = make_function(balanced_config)
    default_info = default_fn.get_kernel_info(1)
    same_default_info = same_default_fn.get_kernel_info(1)
    balanced_info = balanced_fn.get_kernel_info(1)

    assert default_info.file_path == same_default_info.file_path
    assert default_info.file_path != balanced_info.file_path
    assert "_balanced" not in Path(default_info.file_path).stem
    assert Path(balanced_info.file_path).stem.endswith("_balanced")

    default_wrapper = default_fn.instantiate(1)
    default_config.balance_grid = True
    balanced_wrapper = default_fn.instantiate(1)
    assert balanced_wrapper is not default_wrapper
    assert len(default_fn.overloads) == 2
    assert len(default_fn._kernel_info_cache) == 2


@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon",
    reason="Cambricon uses a separate pointwise generator",
)
def test_balanced_grid_real_kernel_covers_non_power_of_two_tail():
    config = CommonCodeGenConfig(
        max_tile_size=4,
        max_grid_size=(3, 1, 1),
        max_num_warps_per_cta=4,
        prefer_block_pointer=False,
        prefer_1d_tile=True,
        balance_grid=True,
    )

    @common_pointwise_dynamic(
        num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=config
    )
    @triton.jit
    def copy_scalar(x):
        return x

    source = torch.arange(15, dtype=torch.float32, device=flag_gems.device)
    actual = copy_scalar(source)
    torch.testing.assert_close(actual, source)


def test_function_schema_with_non_tensor_input():
    schema = FunctionSchema(
        is_tensor=[True, False, True],
        dtypes=[None, float, None],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    assert schema.num_input_tensors() == 2
    assert schema.num_output_tensors() == 1
    assert schema.num_inputs() == 3
    assert schema.num_non_tensor_args() == 1
    assert schema.input_index(0) == 0  # the first input is the first input tensor
    assert schema.input_index(1) == 0  # the second input is the first non tensor input
    assert schema.input_index(2) == 1  # the third input is the second input tensor


def test_function_schema_mismatch_input_num1():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[True, False, True],
            dtypes=[None],
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_mismatch_input_num2():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[True, False, True],
            num_inputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_mismatch_input_num3():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            num_inputs=2,
            dtypes=[None, None, None],
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_missing_output_dtype_promotion_rules():
    with pytest.raises(ValueError):
        schema = FunctionSchema(
            num_inputs=2,
            dtypes=[None, None, None],
        )
        _ = schema


def test_function_schema_mismatch_output_num():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            num_inputs=1,
            num_outputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_missing_input_info():
    with pytest.raises(ValueError):
        schema = FunctionSchema(
            num_outputs=2,
            promotion_methods=[(0, 1, 2, "DEFAULT")],
        )
        _ = schema


def test_function_schema_no_tensor_inputs1():
    # no tensor input is okay with FunctionSchema
    schema = FunctionSchema(
        is_tensor=[False, False, False],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    _ = schema


def test_function_schema_no_tensor_inputs2():
    schema = FunctionSchema(
        num_inputs=3,
        is_tensor=[False, False, False],
        promotion_methods=[(0, 1, 2, "DEFAULT")],
    )
    _ = schema


def test_function_schema_no_outputs1():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[False, False, False],
            promotion_methods=[],
        )
        _ = schema


def test_function_schema_no_outputs2():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(
            is_tensor=[False, False, False],
            num_outputs=0,
            promotion_methods=[],
        )
        _ = schema


def test_function_schema_illegal_dtypes():
    with pytest.raises(AssertionError):
        schema = FunctionSchema(dtypes=[0, False, "a"])
        _ = schema


def test_function_schema_multiple_outputs():
    schema = FunctionSchema(
        num_inputs=3,
        num_outputs=2,
        promotion_methods=[(0, 1, 2, "DEFAULT"), (0, 1, "ALWAYS_BOOL")],
    )
    _ = schema


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_without_non_tensor_args(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device=flag_gems.device)
        y = torch.randn_like(x)
        out = add(x, y)
        torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_non_tensor_args(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device=flag_gems.device)
        y = torch.randn_like(x)
        alpha = 2.0
        out = axpy(x, y, alpha)
        torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_multiple_outputs(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        num_outputs=2,
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def multiple_out(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device=flag_gems.device)
        y = torch.randn_like(x)
        alpha = 2.0
        out0, out1 = multiple_out(x, y, alpha)
        torch.testing.assert_close(out0, alpha * x + y)
        torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_broadcasting(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=True,  # [misaligned address]
    )

    # NOTE: [misaligned address]
    # triton 2.2 may cause Misaligned address when using >=3d tiles in some
    # cases with some zero strides
    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, 1, SIZE], device=flag_gems.device)
    y = torch.randn([1, SIZE, 1], device=flag_gems.device)
    alpha = 2.0
    out = axpy(x, y, alpha)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_broadcasting2(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=True,  # [misaligned address]
    )

    # NOTE: See note [misaligned address]
    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, 1, SIZE], device=flag_gems.device)
    y = torch.randn([], device=flag_gems.device)
    alpha = 2.0
    out = axpy(x, y, alpha)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4108: not working"
)
def test_dynamic_function_with_predefined_out(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([], device=flag_gems.device)
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    out = axpy(x, y, alpha, out0=o)
    torch.testing.assert_close(out, alpha * x + y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4108: not working"
)
def test_dynamic_function_with_some_predefined_out1(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([], device=flag_gems.device)
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4108: not working"
)
def test_dynamic_function_with_some_predefined_out2(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([], device=flag_gems.device)
    alpha = 2.0
    o = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    out0, out1 = axpyaxmy(x, y, alpha, out1=o)
    assert out1 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_bool_input_and_output(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=1,
        is_tensor=[True],
        promotion_methods=[(0, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def invert(x):
        return ~x

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device) > 0
    notx = invert(x)

    torch.testing.assert_close(notx, ~x)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=1,
        is_tensor=[True],
        promotion_methods=[(0, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def invert(x):
        return ~x

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device) > 0
    o = torch.empty_like(x)
    # manually instantiated overload does not handle output allocation
    # since it is kind of low level
    notx = invert.instantiate(3)(x, out0=o)
    torch.testing.assert_close(notx, ~x)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_nd_buffer(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, K], device=flag_gems.device)[::2, ::2, ::2]
    y = torch.randn([N // 2, K // 2, M // 2], device=flag_gems.device).permute(2, 0, 1)
    alpha = 2.0
    o = torch.empty([M // 2, N // 2, K // 2], device=flag_gems.device)
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_nd_buffer_out_permute(use_1d_tile, use_block_pointer):
    if flag_gems.vendor_name != "cambricon":
        return

    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, K], device="cuda")[::2, ::2, ::2]
    y = torch.randn([M // 2, N // 2, K // 2], device="cuda")
    alpha = 2.0
    o = torch.empty([M // 2, K // 2, N // 2], device="cuda").permute(0, 2, 1)
    o2 = torch.empty([K // 2, M // 2, N // 2], device="cuda").permute(1, 2, 0)
    print(o.stride(), o2.stride())
    out0, out1 = axpyaxmy(x, y, alpha, out0=o, out1=o2)
    assert out0 is o and out1 is o2
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_nd_buffer_broadcast(use_1d_tile, use_block_pointer):
    if flag_gems.vendor_name != "cambricon":
        return

    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, 2], device="cuda")[::2, ::2, ::2]
    y = torch.randn([1, K // 2, M // 2], device="cuda").permute(2, 0, 1)
    alpha = 2.0
    o = torch.empty([M // 2, N // 2, K // 2], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_nd_buffer_expand(use_1d_tile, use_block_pointer):
    if flag_gems.vendor_name != "cambricon":
        return

    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = (
        torch.randn([1, K // 2, N // 2], device="cuda")
        .permute(0, 2, 1)
        .expand([M // 2, N // 2, K // 2])
    )
    y = (
        torch.randn([1, K // 2, M // 2], device="cuda")
        .permute(2, 0, 1)
        .expand([M // 2, N // 2, K // 2])
    )
    alpha = 2.0
    o = torch.empty([M // 2, N // 2, K // 2], device="cuda")
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


# Cambricon add end.


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_different_stride_order(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    M, N, K = 40, 60, 80
    x = torch.randn([M, N, K], device=flag_gems.device)
    y = torch.randn([N, K, M], device=flag_gems.device).permute(2, 0, 1)
    alpha = 2.0
    o = torch.empty([M, N, K], device=flag_gems.device)
    out0, out1 = axpyaxmy(x, y, alpha, out0=o)
    assert out0 is o
    torch.testing.assert_close(out0, alpha * x + y)
    torch.testing.assert_close(out1, alpha * x - y)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_mixing_strided_buffer_and_tensor(
    use_block_pointer,
):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device))
    out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)

    assert isinstance(out0, torch.Tensor)
    assert isinstance(out1, StridedBuffer)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_support_broadcasting1(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([1, SIZE], device=flag_gems.device)
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device))

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_support_broadcasting2(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([SIZE, 1, SIZE], device=flag_gems.device)
    alpha = 2.0
    _out0 = torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device)
    _out1 = StridedBuffer(torch.empty([SIZE, SIZE, SIZE], device=flag_gems.device))

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha, out0=_out0, out1=_out1)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_manual_instantiation_does_not_allocate_output(
    use_block_pointer,
):
    # manually instantiated overload does not support broadcasting of operands
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT"), (0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpyaxmy(x, y, alpha):
        return alpha * x + y, alpha * x - y

    SIZE = 10
    x = torch.randn([SIZE, SIZE, SIZE], device=flag_gems.device)
    y = torch.randn([SIZE, 1, SIZE], device=flag_gems.device)
    alpha = 2.0

    with pytest.raises(Exception):
        out0, out1 = axpyaxmy.instantiate(3)(x, y, alpha)


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_gsl(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=512,
        max_grid_size=(80, 1, 1),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    SIZE = 2
    for ndim in range(8):
        shape = [SIZE] * ndim
        x = torch.randn(shape, device=flag_gems.device)
        y = torch.randn_like(x)
        out = add(x, y)
        torch.testing.assert_close(out, x + y)


@pytest.mark.skipif(
    get_device_properties(0).total_memory < (80 * 1024**3),
    reason="This test requires a lot of memory.",
)
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_int64_index(use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(MAX_GRID_SIZE_X, 1, 1),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=config)
    @triton.jit
    def f(x):
        return x * 2.0

    x = torch.randn((2, 1024, 1024, 1024), dtype=torch.float16, device=flag_gems.device)
    y1 = f(x)
    y2 = x * 2.0
    torch.testing.assert_close(y1, y2)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_0d_task(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def add(x, y):
        return x + y

    shape = ()
    x = torch.randn(shape, device=flag_gems.device)
    y = torch.randn_like(x)
    out = add(x, y)
    torch.testing.assert_close(out, x + y)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Isue #2837: AssertionError"
)
def test_dynamic_function_zero_sized_task_unary(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(num_inputs=1, promotion_methods=[(0, "DEFAULT")], config=config)
    @triton.jit
    def f(x):
        return x * 2.0

    shape = (0, 10)
    x = torch.randn(shape, device=flag_gems.device)
    out = f(x)
    torch.testing.assert_close(out, x * 2.0)


@pytest.mark.parametrize("use_1d_tile", [True, False])
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads", reason="Issue #2837: AssertionError"
)
def test_dynamic_function_zero_sized_task_binary(use_1d_tile, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=(65536, 65536, 65536),
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=use_1d_tile,
    )

    @pointwise_dynamic(
        num_inputs=2, promotion_methods=[(0, 1, "DEFAULT")], config=config
    )
    @triton.jit
    def f(x, y):
        return x * 2.0 + y

    shape = (0, 10)
    x = torch.randn(shape, device=flag_gems.device)
    y = torch.randn_like(x)
    out = f(x, y)
    torch.testing.assert_close(out, x * 2.0 + y)


def f_for_concurrency_test(x, alpha, use_block_pointer):
    config = CodeGenConfig(
        max_tile_size=1024,
        max_grid_size=MAX_GRID_SIZES,
        max_num_warps_per_cta=32,
        prefer_block_pointer=use_block_pointer,
        prefer_1d_tile=False,
    )

    @pointwise_dynamic(
        num_inputs=3,
        is_tensor=[True, True, False],
        promotion_methods=[(0, 1, "DEFAULT")],
        config=config,
    )
    @triton.jit
    def axpy(x, y, alpha):
        return alpha * x + y

    y = torch.zeros_like(x)
    out = axpy(x, y, alpha)
    return out


@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_multithread(use_block_pointer):
    shape = [128]
    alpha = 2.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        inputs = [torch.randn(shape, device=flag_gems.device) for _ in range(32)]
        expected_outs = [item * alpha for item in inputs]
        outs = []
        for item in inputs:
            out_future = executor.submit(
                f_for_concurrency_test, item, alpha, use_block_pointer
            )
            outs.append(out_future)
        outs = [item.result() for item in outs]

    for out, expected_out in zip(outs, expected_outs):
        torch.testing.assert_close(out, expected_out)


@pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise",
    reason="Issues #3837: spawn not support ptpu tensor",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #4110: not working",
)
@pytest.mark.parametrize("use_block_pointer", USE_BLOCK_POINTER)
def test_dynamic_function_with_multiprocess(use_block_pointer):
    shape = [128]
    alpha = 2.0
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=ctx
    ) as executor:
        inputs = [torch.randn(shape, device=flag_gems.device) for _ in range(32)]
        expected_outs = [item * alpha for item in inputs]
        outs = []
        for item in inputs:
            out_future = executor.submit(
                f_for_concurrency_test, item, alpha, use_block_pointer
            )
            outs.append(out_future)
        outs = [item.result() for item in outs]

        for out, expected_out in zip(outs, expected_outs):
            torch.testing.assert_close(out, expected_out)


# Complex number tests

COMPLEX_DTYPES = [torch.complex64, torch.complex128]


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_elementwise_tensor_tensor(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_func(x, y, alpha):
        return x + y * alpha

    add_func.register_complex(mode=ComplexMode.ELEMENTWISE)

    SIZE = 2
    for ndim in range(1, 5):
        shape = [SIZE] * ndim
        a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        b = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        alpha = 2.0
        out = add_func(a, b, alpha)
        torch.testing.assert_close(out, a + b * alpha)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_elementwise_tensor_scalar(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_tt(x, y, alpha):
        return x + y * alpha

    @pointwise_dynamic(
        is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_ts(x, y, alpha):
        return x + y * alpha

    add_tt.register_complex(mode=ComplexMode.ELEMENTWISE)
    add_ts.register_complex(
        mode=ComplexMode.ELEMENTWISE,
        tensorize_scalars=True,
        fallback_target=add_tt,
    )

    shape = [64]
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    b = 1.5 + 2.0j
    alpha = 1.0
    out = add_ts(a, b, alpha)
    torch.testing.assert_close(out, a + b * alpha)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_elementwise_broadcast(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_func(x, y, alpha):
        return x + y * alpha

    add_func.register_complex(mode=ComplexMode.ELEMENTWISE)

    a = torch.randn([4, 16], dtype=dtype, device=flag_gems.device)
    b = torch.randn([16], dtype=dtype, device=flag_gems.device)
    alpha = 1.0
    out = add_func(a, b, alpha)
    torch.testing.assert_close(out, a + b * alpha)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_elementwise_mixed_real_complex(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_func(x, y, alpha):
        return x + y * alpha

    add_func.register_complex(mode=ComplexMode.ELEMENTWISE)

    shape = [128]
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    b = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    alpha = 1.0
    out = add_func(a, b, alpha)
    torch.testing.assert_close(out, a + b * alpha)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_cross_tensor_tensor(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, True, True],
        num_outputs=2,
        promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
    )
    @triton.jit
    def mul_cross(ar, ai, br, bi):
        real = ar * br - ai * bi
        imag = ar * bi + ai * br
        return real, imag

    @pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
    @triton.jit
    def mul_func(x, y):
        return x * y

    mul_func.register_complex(mode=ComplexMode.CROSS, cross_kernel=mul_cross)

    SIZE = 2
    for ndim in range(1, 5):
        shape = [SIZE] * ndim
        a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        b = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        out = mul_func(a, b)
        torch.testing.assert_close(out, a * b)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_cross_tensor_scalar(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, True, True],
        num_outputs=2,
        promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
    )
    @triton.jit
    def mul_cross(ar, ai, br, bi):
        real = ar * br - ai * bi
        imag = ar * bi + ai * br
        return real, imag

    @pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
    @triton.jit
    def mul_tt(x, y):
        return x * y

    @pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
    @triton.jit
    def mul_ts(x, y):
        return x * y

    mul_tt.register_complex(mode=ComplexMode.CROSS, cross_kernel=mul_cross)
    mul_ts.register_complex(
        mode=ComplexMode.CROSS,
        tensorize_scalars=True,
        fallback_target=mul_tt,
    )

    shape = [64]
    a = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    b = 2.0 + 3.0j
    out = mul_ts(a, b)
    torch.testing.assert_close(out, a * b)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_cross_broadcast(dtype):
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, True, True],
        num_outputs=2,
        promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
    )
    @triton.jit
    def mul_cross(ar, ai, br, bi):
        real = ar * br - ai * bi
        imag = ar * bi + ai * br
        return real, imag

    @pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
    @triton.jit
    def mul_func(x, y):
        return x * y

    mul_func.register_complex(mode=ComplexMode.CROSS, cross_kernel=mul_cross)

    a = torch.randn([4, 16], dtype=dtype, device=flag_gems.device)
    b = torch.randn([16], dtype=dtype, device=flag_gems.device)
    out = mul_func(a, b)
    torch.testing.assert_close(out, a * b)


@pytest.mark.parametrize("dtype", COMPLEX_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
def test_complex_real_inputs_bypass(dtype):
    """When all inputs are real, complex-registered kernel should still work."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.complex128:
        pytest.skip("Issue #5253: Not supported")

    @pointwise_dynamic(
        is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")]
    )
    @triton.jit
    def add_func(x, y, alpha):
        return x + y * alpha

    add_func.register_complex(mode=ComplexMode.ELEMENTWISE)

    shape = [128]
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    a = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    b = torch.randn(shape, dtype=real_dtype, device=flag_gems.device)
    alpha = 1.0
    out = add_func(a, b, alpha)
    torch.testing.assert_close(out, a + b * alpha)
