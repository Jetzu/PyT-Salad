# Topologically Sorted Source Nodes: [cond_nbr_stack], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   cond_nbr_stack => cat_11
# Graph fragment:
#   %cat_11 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_5, %cat_6, %cat_7, %cat_8, %cat_9, %cat_10],), kwargs = {})
triton_poi_fused_stack_72 = async_compile.triton('triton_poi_fused_stack_72', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i16', 2: '*fp32', 3: '*i16', 4: '*fp32', 5: '*i16', 6: '*fp32', 7: '*i16', 8: '*fp32', 9: '*i16', 10: '*fp32', 11: '*i16', 12: '*fp32', 13: '*i64', 14: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_72', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'ACB470BEFDE58A90F00079B5434325E87B6B4313CA4901D1EAFBC23641D24150', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 48)
    x0 = xindex % 48
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tmp5 >= tmp1
    tmp7 = tl.full([1], 32, tl.int64)
    tmp8 = tmp5 < tmp7
    tmp9 = tmp8 & tmp4
    tmp10 = tl.load(in_ptr0 + ((32*x1) + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp5 >= tmp7
    tmp12 = tl.full([1], 48, tl.int64)
    tmp13 = tmp5 < tmp12
    tmp14 = tmp11 & tmp4
    tl.device_assert((x1 < 100) | ~(tmp14 & xmask), "index out of bounds: x1 < 100")
    tmp16 = tl.load(in_ptr1 + ((100*x1) + ((-32) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16.to(tl.int64)
    tmp18 = tl.full([XBLOCK], 100, tl.int32)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp17 < 0
    tmp21 = tl.where(tmp20, tmp19, tmp17)
    tl.device_assert(((0 <= tl.broadcast_to(tmp21, [XBLOCK])) & (tl.broadcast_to(tmp21, [XBLOCK]) < 100)) | ~(tmp14 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp21, [XBLOCK]) < 100")
    tmp23 = tl.load(in_ptr2 + (tmp21 + (100*x1)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = float("inf")
    tmp25 = tmp23 < tmp24
    tmp26 = tl.full([1], -1, tl.int64)
    tmp27 = tl.where(tmp25, tmp17, tmp26)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp14, tmp27, tmp28)
    tmp30 = tl.where(tmp8, tmp10, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp4, tmp30, tmp31)
    tmp33 = tmp0 >= tmp3
    tmp34 = tl.full([1], 200, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tmp33 & tmp35
    tmp37 = tmp8 & tmp36
    tmp38 = tl.load(in_ptr0 + ((32*((-100) + x1)) + x0), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp11 & tmp36
    tl.device_assert((tl.where(((-100) + x1) < 0, 100 + ((-100) + x1), (-100) + x1) < 100) | ~(tmp39 & xmask), "index out of bounds: tl.where(((-100) + x1) < 0, 100 + ((-100) + x1), (-100) + x1) < 100")
    tmp41 = tl.load(in_ptr3 + ((100*((-100) + x1)) + ((-32) + x0)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41.to(tl.int64)
    tmp43 = tmp42 + tmp18
    tmp44 = tmp42 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp42)
    tl.device_assert(((0 <= tl.broadcast_to(tmp45, [XBLOCK])) & (tl.broadcast_to(tmp45, [XBLOCK]) < 100)) | ~(tmp39 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp45, [XBLOCK]) < 100")
    tmp47 = tl.load(in_ptr4 + (tmp45 + (100*(tl.where(((-100) + x1) < 0, 100 + ((-100) + x1), (-100) + x1)))), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 < tmp24
    tmp49 = tl.where(tmp48, tmp42, tmp26)
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp39, tmp49, tmp50)
    tmp52 = tl.where(tmp8, tmp38, tmp51)
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp36, tmp52, tmp53)
    tmp55 = tmp0 >= tmp34
    tmp56 = tl.full([1], 300, tl.int64)
    tmp57 = tmp0 < tmp56
    tmp58 = tmp55 & tmp57
    tmp59 = tmp8 & tmp58
    tmp60 = tl.load(in_ptr0 + ((32*((-200) + x1)) + x0), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp11 & tmp58
    tl.device_assert(((0 <= tl.where(((-200) + x1) < 0, 100 + ((-200) + x1), (-200) + x1)) & (tl.where(((-200) + x1) < 0, 100 + ((-200) + x1), (-200) + x1) < 100)) | ~(tmp61 & xmask), "index out of bounds: 0 <= tl.where(((-200) + x1) < 0, 100 + ((-200) + x1), (-200) + x1) < 100")
    tmp63 = tl.load(in_ptr5 + ((100*((-200) + x1)) + ((-32) + x0)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp63.to(tl.int64)
    tmp65 = tmp64 + tmp18
    tmp66 = tmp64 < 0
    tmp67 = tl.where(tmp66, tmp65, tmp64)
    tl.device_assert(((0 <= tl.broadcast_to(tmp67, [XBLOCK])) & (tl.broadcast_to(tmp67, [XBLOCK]) < 100)) | ~(tmp61 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp67, [XBLOCK]) < 100")
    tmp69 = tl.load(in_ptr6 + (tmp67 + (100*(tl.where(((-200) + x1) < 0, 100 + ((-200) + x1), (-200) + x1)))), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 < tmp24
    tmp71 = tl.where(tmp70, tmp64, tmp26)
    tmp72 = tl.full(tmp71.shape, 0.0, tmp71.dtype)
    tmp73 = tl.where(tmp61, tmp71, tmp72)
    tmp74 = tl.where(tmp8, tmp60, tmp73)
    tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
    tmp76 = tl.where(tmp58, tmp74, tmp75)
    tmp77 = tmp0 >= tmp56
    tmp78 = tl.full([1], 400, tl.int64)
    tmp79 = tmp0 < tmp78
    tmp80 = tmp77 & tmp79
    tmp81 = tmp8 & tmp80
    tmp82 = tl.load(in_ptr0 + ((32*((-300) + x1)) + x0), tmp81 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp11 & tmp80
    tl.device_assert(((0 <= tl.where(((-300) + x1) < 0, 100 + ((-300) + x1), (-300) + x1)) & (tl.where(((-300) + x1) < 0, 100 + ((-300) + x1), (-300) + x1) < 100)) | ~(tmp83 & xmask), "index out of bounds: 0 <= tl.where(((-300) + x1) < 0, 100 + ((-300) + x1), (-300) + x1) < 100")
    tmp85 = tl.load(in_ptr7 + ((100*((-300) + x1)) + ((-32) + x0)), tmp83 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp85.to(tl.int64)
    tmp87 = tmp86 + tmp18
    tmp88 = tmp86 < 0
    tmp89 = tl.where(tmp88, tmp87, tmp86)
    tl.device_assert(((0 <= tl.broadcast_to(tmp89, [XBLOCK])) & (tl.broadcast_to(tmp89, [XBLOCK]) < 100)) | ~(tmp83 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp89, [XBLOCK]) < 100")
    tmp91 = tl.load(in_ptr8 + (tmp89 + (100*(tl.where(((-300) + x1) < 0, 100 + ((-300) + x1), (-300) + x1)))), tmp83 & xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp91 < tmp24
    tmp93 = tl.where(tmp92, tmp86, tmp26)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp83, tmp93, tmp94)
    tmp96 = tl.where(tmp8, tmp82, tmp95)
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp80, tmp96, tmp97)
    tmp99 = tmp0 >= tmp78
    tmp100 = tl.full([1], 500, tl.int64)
    tmp101 = tmp0 < tmp100
    tmp102 = tmp99 & tmp101
    tmp103 = tmp8 & tmp102
    tmp104 = tl.load(in_ptr0 + ((32*((-400) + x1)) + x0), tmp103 & xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp11 & tmp102
    tl.device_assert(((0 <= tl.where(((-400) + x1) < 0, 100 + ((-400) + x1), (-400) + x1)) & (tl.where(((-400) + x1) < 0, 100 + ((-400) + x1), (-400) + x1) < 100)) | ~(tmp105 & xmask), "index out of bounds: 0 <= tl.where(((-400) + x1) < 0, 100 + ((-400) + x1), (-400) + x1) < 100")
    tmp107 = tl.load(in_ptr9 + ((100*((-400) + x1)) + ((-32) + x0)), tmp105 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp107.to(tl.int64)
    tmp109 = tmp108 + tmp18
    tmp110 = tmp108 < 0
    tmp111 = tl.where(tmp110, tmp109, tmp108)
    tl.device_assert(((0 <= tl.broadcast_to(tmp111, [XBLOCK])) & (tl.broadcast_to(tmp111, [XBLOCK]) < 100)) | ~(tmp105 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp111, [XBLOCK]) < 100")
    tmp113 = tl.load(in_ptr10 + (tmp111 + (100*(tl.where(((-400) + x1) < 0, 100 + ((-400) + x1), (-400) + x1)))), tmp105 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp113 < tmp24
    tmp115 = tl.where(tmp114, tmp108, tmp26)
    tmp116 = tl.full(tmp115.shape, 0.0, tmp115.dtype)
    tmp117 = tl.where(tmp105, tmp115, tmp116)
    tmp118 = tl.where(tmp8, tmp104, tmp117)
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp102, tmp118, tmp119)
    tmp121 = tmp0 >= tmp100
    tmp122 = tl.full([1], 600, tl.int64)
    tmp123 = tmp0 < tmp122
    tmp124 = tmp8 & tmp121
    tmp125 = tl.load(in_ptr0 + ((32*((-500) + x1)) + x0), tmp124 & xmask, eviction_policy='evict_last', other=0.0)
    tmp126 = tmp11 & tmp121
    tl.device_assert((0 <= tl.where(((-500) + x1) < 0, 100 + ((-500) + x1), (-500) + x1)) | ~(tmp126 & xmask), "index out of bounds: 0 <= tl.where(((-500) + x1) < 0, 100 + ((-500) + x1), (-500) + x1)")
    tmp128 = tl.load(in_ptr11 + ((100*((-500) + x1)) + ((-32) + x0)), tmp126 & xmask, eviction_policy='evict_last', other=0.0)
    tmp129 = tmp128.to(tl.int64)
    tmp130 = tmp129 + tmp18
    tmp131 = tmp129 < 0
    tmp132 = tl.where(tmp131, tmp130, tmp129)
    tl.device_assert(((0 <= tl.broadcast_to(tmp132, [XBLOCK])) & (tl.broadcast_to(tmp132, [XBLOCK]) < 100)) | ~(tmp126 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp132, [XBLOCK]) < 100")
    tmp134 = tl.load(in_ptr12 + (tmp132 + (100*(tl.where(((-500) + x1) < 0, 100 + ((-500) + x1), (-500) + x1)))), tmp126 & xmask, eviction_policy='evict_last', other=0.0)
    tmp135 = tmp134 < tmp24
    tmp136 = tl.where(tmp135, tmp129, tmp26)
    tmp137 = tl.full(tmp136.shape, 0.0, tmp136.dtype)
    tmp138 = tl.where(tmp126, tmp136, tmp137)
    tmp139 = tl.where(tmp8, tmp125, tmp138)
    tmp140 = tl.full(tmp139.shape, 0.0, tmp139.dtype)
    tmp141 = tl.where(tmp121, tmp139, tmp140)
    tmp142 = tl.where(tmp102, tmp120, tmp141)
    tmp143 = tl.where(tmp80, tmp98, tmp142)
    tmp144 = tl.where(tmp58, tmp76, tmp143)
    tmp145 = tl.where(tmp36, tmp54, tmp144)
    tmp146 = tl.where(tmp4, tmp32, tmp145)
    tl.store(out_ptr0 + (x2), tmp146, xmask)
''', device_str='cuda')


# kernel path: /tmp/907104.helma/n026pair/inductor/tmpbs8nlihi/lk/clkv35ef3lw5g3ysymiz6kvo5gdloh2u7o7cwkiqtveylp7mi7q5.py

# Topologically Sorted Source Nodes: [getitem_143, same_batch_1, getitem_141, same_chain_1, resi_t, getitem_145, rel, clamp_5, binned, other, binned_1, binned_2], Original ATen: [aten.index, aten.eq, aten.repeat, aten.sub, aten.clamp, aten.add, aten.full_like, aten.where]
# Source node to ATen node mapping:
#   binned => add_27
#   binned_1 => where_33
#   binned_2 => where_34
#   clamp_5 => clamp_max, clamp_min_5
#   getitem_141 => index_28
#   getitem_143 => index_29
#   getitem_145 => index_30
#   other => full_default_32
#   rel => sub_19
#   resi_t => repeat_1
#   same_batch_1 => eq_11
#   same_chain_1 => eq_10
# Graph fragment:
#   %index_29 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_17, [%view_15]), kwargs = {})
#   %eq_11 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_97, %index_29), kwargs = {})
#   %index_28 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%view_16, [%view_15]), kwargs = {})
#   %eq_10 : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze_96, %index_28), kwargs = {})
#   %repeat_1 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%arg16_1, [6]), kwargs = {})
#   %index_30 : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%repeat_1, [%view_15]), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_98, %index_30), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_19, -32), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 32), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, 32), kwargs = {})
#   %full_default_32 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([600, 48], 65), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_33 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_10, %add_27, %full_default_32), kwargs = {})
#   %where_34 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq_11, %where_33, %full_default_32), kwargs = {})
triton_poi_fused_add_clamp_eq_full_like_index_repeat_sub_where_74 = async_compile.triton('triton_poi_fused_add_clamp_eq_full_like_index_repeat_sub_where_74', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[32768], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: '*i64', 4: '*i64', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_eq_full_like_index_repeat_sub_where_74', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ACB470BEFDE58A90F00079B5434325E87B6B4313CA4901D1EAFBC23641D24150', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_eq_full_like_index_repeat_sub_where_74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 48)
    tmp0 = tl.load(in_ptr0 + ((x2 // 48) % 100), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp18 = tl.load(in_ptr2 + ((x2 // 48) % 100), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + ((x2 // 48) % 100), xmask, eviction_policy='evict_last')
    tmp1 = 100*(x1 // 100)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tmp3 + tmp1
    tmp7 = 99 + (100*(x1 // 100))
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.full([XBLOCK], 600, tl.int32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp8 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp8)
    tl.device_assert(((0 <= tmp12) & (tmp12 < 600)) | ~(xmask), "index out of bounds: 0 <= tmp12 < 600")
    tmp14 = tl.load(in_ptr0 + (tmp12 % 100), xmask, eviction_policy='evict_last')
    tmp15 = 100*((tmp12 // 100) % 6)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp2 == tmp16
    tmp19 = tmp18 + tmp1
    tmp20 = tl.load(in_ptr2 + (tmp12 % 100), xmask, eviction_policy='evict_last')
    tmp21 = tmp20 + tmp15
    tmp22 = tmp19 == tmp21
    tmp24 = tl.load(in_ptr3 + (tmp12 % 100), xmask, eviction_policy='evict_last')
    tmp25 = tmp23 - tmp24
    tmp26 = tl.full([1], -32, tl.int64)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp28 = tl.full([1], 32, tl.int64)
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp29 + tmp28
    tmp31 = tl.full([1], 65, tl.int64)
    tmp32 = tl.where(tmp22, tmp30, tmp31)
    tmp33 = tl.where(tmp17, tmp32, tmp31)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: /tmp/907104.helma/n026pair/inductor/tmpbs8nlihi/ja/cjakbeqxrstviw3vd37d473gazajioccqd7d77cnmbjqnwqceuz5.py
