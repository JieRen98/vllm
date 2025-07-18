import torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
import numpy as np
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_zero_points


def w_round(x):
    return torch.floor(x + 0.5)


def decode(
        w: torch.Tensor,
        w_scale,
        w_code_scale,
        w_code_zp,
        w_super_scale=None,
):
    # step0: w dtype: int8, shape: [num_experts, in_feature_size // pack_num, out_feature_size]
    # where pack_num = 4
    pack_num = 4
    bzp = 32
    num_experts, pack_in_feature_size, out_feature_size = w.shape

    in_feature_size = pack_in_feature_size * pack_num
    # step1: w need to unzip to shape: [num_experts, in_feature_size, out_feature_size]
    # here we use broadcast operation to implcitly expand the last dimension

    w = w.permute(dims=[0, 2, 1]).reshape([num_experts, out_feature_size, pack_in_feature_size, 1])

    # for support repeat_interleave, w cast to int32
    w = w.to(torch.int32)
    w = w.repeat_interleave(pack_num, dim=-1)
    w = w.reshape([num_experts, out_feature_size, in_feature_size])
    w = w.permute(dims=[0, 2, 1])

    # step2: w need to first dequant
    # w_code_scale shape: [num_experts, out_feature_size]
    # w_code_zp shape: [num_experts, out_feature_size]
    w_code_scale = w_code_scale.reshape([num_experts, 1, out_feature_size])
    w_code_zp = w_code_zp.reshape([num_experts, 1, out_feature_size])

    # w = torch.addcmul(w_code_zp, 1, w.to(torch.float32), w_code_scale).to(torch.int16)
    w = w_round(w.to(torch.float32) * w_code_scale + w_code_zp).to(torch.int16)

    # step3: w need to shifted and mask the original weight to unzip
    bit_shift = torch.tensor([9, 6, 3, 0], dtype=torch.int16, device=w.device)
    in_feature_bit_shift = bit_shift[torch.arange(in_feature_size) % pack_num]
    in_feature_bit_shift = in_feature_bit_shift.reshape([1, in_feature_size, 1])
    mask = torch.tensor(0x3F, dtype=torch.int16, device=w.device)

    # step4: w need to shift and mask and second dequant
    w = ((w >> in_feature_bit_shift) & mask).to(w_scale.dtype)

    perm_one = [0, 1, 16, 17, 32, 33, 48, 49, 2, 3, 18, 19, 34, 35, 50, 51]

    perm = []
    for i in range(4):
        perm.extend(list(4 * i + p for p in perm_one))

    assert perm == list(4 * i + p for i in range(4) for p in perm_one)

    assert len(perm) == 4 * 4 * 4

    assert in_feature_size % len(perm) == 0

    perm = torch.tensor(perm, dtype=torch.int, device=w.device)

    w = w.reshape([num_experts, in_feature_size // perm.numel(), perm.numel(), out_feature_size])
    w = w[:, :, perm, :]

    w = w.reshape([num_experts, in_feature_size, out_feature_size])

    if w_super_scale is not None:
        # w_super_scale shape: [num_experts, out_feature_size]
        # w_scale shape: [num_experts, in_feature_size // group_size,out_feature_size]
        # group_size = 64
        w_super_scale = w_super_scale.reshape([num_experts, 1, out_feature_size])
        w_scale = w_scale * w_super_scale

    # w_scale reshape to [num_experts, in_feature_size, out_feature_size]
    group_size = 64
    w_scale = w_scale.reshape([num_experts, in_feature_size // group_size, 1, out_feature_size])
    w_scale = w_scale.repeat_interleave(group_size, dim=2).reshape([num_experts, in_feature_size, out_feature_size])

    w = (w - bzp).to(w_scale.dtype) * w_scale

    return w


def get_perm():
    block_cols = 8 * 2 * 4

    perm = []

    for row_group in range(4):
        for tid in range(32):
            row = tid % 4
            col = tid // 4

            for col_group in range(8):
                perm.append(col + col_group * 8 + (row * 4 + row_group) * block_cols)

    return torch.tensor(perm, dtype=torch.int32)


def perm_and_pack_weight(w: torch.Tensor):
    assert w.dim() == 2
    assert w.shape[0] % 16 == 0
    assert w.shape[1] % 64 == 0
    assert w.dtype == torch.int8 or w.dtype == torch.uint8
    k, n = w.shape

    w_reshaped = w.reshape((k // 16, 16, n // 64, 64))
    w_reshaped = w_reshaped.permute((0, 2, 1, 3))
    perm = get_perm().to(w.device)

    assert perm.numel() == 16 * 64

    w_reshaped = w_reshaped.reshape((-1, perm.numel()))[:, perm.view(-1)]

    w_reshaped = w_reshaped.reshape(k // 16, n // 64, 4, 32, 8)

    w_reshaped = w_reshaped.permute((0, 2, 1, 3, 4))
    assert w_reshaped.shape == (k // 16, 4, n // 64, 32, 8)

    w_reshaped = w_reshaped.reshape(k // 4, n * 4)

    w_reshaped = w_reshaped.cpu().numpy().astype(np.uint16)

    w_packed = np.zeros((k // 4, n * 4 // 2), dtype=np.uint16)

    pack_factor = 16 // 8
    for i in range(pack_factor):
        w_packed |= w_reshaped[:, i::pack_factor] << 8 * i

    return torch.from_numpy(w_packed).to(w.device)


def convert_uint8_to_uint32(tensor_uint8):
    # 分离高4位和低4位
    high_bits = (tensor_uint8 >> 4 & 0x0F)  # 提取高4位
    low_bits = (tensor_uint8 & 0x0F)  # 提取低4位（与0x0F进行与操作）

    # 沿新维度堆叠并重塑形状
    combined = torch.stack((high_bits, low_bits), dim=2)  # 形状变为 (a, b, 2, c)
    result = combined.view(tensor_uint8.size(0), tensor_uint8.size(1) * 2, tensor_uint8.size(2))  # 重塑为 (a, 2b, c)

    # 转换为uint32类型
    return result.to(torch.int32)


# 示例用法
if __name__ == "__main__":
    GPTQ_MARLIN_TILE = 16
    GROUP_SIZE = 64

    device = torch.device("cuda:0")

    dtype = torch.bfloat16
    # dtype = torch.float16
    quant_type = scalar_types.uint4

    m = 128
    n = 3584
    k = 8192
    e = 64
    topk = 8

    path = '/home/jie/Repos/marlin_int2/'

    qweight1 = torch.load(path + 'ffn1_weight', map_location=device)
    weight_scales1 = torch.load(path + 'ffn1_weight_scale', map_location=device)
    super_scales1 = torch.load(path + 'ffn1_super_scales', map_location=device).to(dtype)
    code_scales1 = torch.load(path + 'ffn1_code_scale', map_location=device)
    code_zeros1 = torch.load(path + 'ffn1_code_zp', map_location=device)

    qweight2 = torch.load(path + 'ffn2_weight', map_location=device)
    weight_scales2 = torch.load(path + 'ffn2_weight_scale', map_location=device)
    super_scales2 = torch.load(path + 'ffn2_super_scales', map_location=device).to(dtype)
    code_scales2 = torch.load(path + 'ffn2_code_scale', map_location=device)
    code_zeros2 = torch.load(path + 'ffn2_code_zp', map_location=device)

    gate_weight = torch.load(path + 'gate_weight', map_location=device)

    fused_moe_out = torch.load(path + 'fused_moe_out', map_location=device)

    a = torch.load(path + 'x', map_location=device).to(dtype)

    # a = torch.zeros_like(a)
    # d = torch.diagonal(torch.ones_like(a))
    # a[:len(d), :len(d)] = torch.diag(d)
    # a[16, 16] = 1

    gate_out = torch.matmul(a.to(torch.float32), gate_weight)

    scores = F.softmax(gate_out, dim=-1)
    topk_weights, topk_ids, _ = fused_topk(a, scores, topk, True)

    gate_correction_bias = np.load(path + 'gate_correction_bias.npy')
    gate_correction_bias = torch.from_numpy(gate_correction_bias).to(device)

    # e_idx = 0
    # qweight1 = qweight1[e_idx].unsqueeze(0)
    # weight_scales1 = weight_scales1[e_idx].unsqueeze(0)
    # super_scales1 = super_scales1[e_idx].unsqueeze(0)
    # code_scales1 = code_scales1[e_idx].unsqueeze(0)
    # code_zeros1 = code_zeros1[e_idx].unsqueeze(0)
    # qweight2 = qweight2[e_idx].unsqueeze(0)
    # weight_scales2 = weight_scales2[e_idx].unsqueeze(0)
    # super_scales2 = super_scales2[e_idx].unsqueeze(0)
    # code_scales2 = code_scales2[e_idx].unsqueeze(0)
    # code_zeros2 = code_zeros2[e_idx].unsqueeze(0)
    #
    # gate_out = gate_out.sum(dim=1, keepdim=True)
    # gate_correction_bias = gate_correction_bias.sum(dim=1, keepdim=True)
    # gate_correction_bias = torch.zeros_like(gate_correction_bias)
    #
    # topk = 1
    # e = 1

    # scores = F.softmax(gate_out, dim=-1)
    # _, topk_ids = torch.topk(gate_correction_bias + scores, k=topk, dim=-1)
    # topk_weights, _ = torch.topk(scores, k=topk, dim=-1)
    # topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)


    def find_unpack(i):
        e_qweight = qweight1[i]
        e_qweight.unsqueeze_(0)
        e_weight_scale = weight_scales1[i]
        e_weight_scale.unsqueeze_(0)
        e_weight_scale = convert_uint8_to_uint32(e_weight_scale)
        e_code_scale = code_scales1[i]
        e_code_scale.unsqueeze_(0)
        e_code_zeros = code_zeros1[i]
        e_code_zeros.unsqueeze_(0)
        e_super_scales1 = super_scales1[i]
        e_super_scales1.unsqueeze_(0)

        # e_code_scale = torch.ones_like(e_code_scale)
        # e_code_zeros = torch.zeros_like(e_code_zeros)
        # e_weight_scale = torch.ones_like(e_weight_scale)
        # e_super_scales1 = torch.ones_like(e_super_scales1)

        return decode(e_qweight, e_weight_scale, e_code_scale, e_code_zeros, e_super_scales1)


    # e0_weight = find_unpack(0)
    #
    # fused_moe_out_fp32 = torch.matmul(a.float(), e0_weight[0].float()).to(dtype)
    # fused_moe_out = torch.matmul(a, e0_weight[0])

    qweight1_new = []
    for dim in range(qweight1.shape[0]):
        qweight1_new.append(perm_and_pack_weight(qweight1[dim]))

    qweight1 = torch.stack(qweight1_new)
    del qweight1_new

    qweight2_new = []
    for dim in range(qweight2.shape[0]):
        qweight2_new.append(perm_and_pack_weight(qweight2[dim]))

    qweight2 = torch.stack(qweight2_new)
    del qweight2_new

    weight_scales1 = convert_uint8_to_uint32(weight_scales1)

    # code_scales1 = torch.ones_like(code_scales1)
    # code_zeros1 = torch.zeros_like(code_zeros1)
    # weight_scales1 = torch.ones_like(weight_scales1)
    # super_scales1 = torch.ones_like(super_scales1)

    weight_scales1_new = []
    for dim in range(weight_scales1.shape[0]):
        weight_scales1_new.append(marlin_zero_points(weight_scales1[dim], k // GROUP_SIZE, 2 * n, 4))

    weight_scales1 = torch.stack(weight_scales1_new)
    del weight_scales1_new

    weight_scales2 = convert_uint8_to_uint32(weight_scales2)
    weight_scales2_new = []
    for dim in range(weight_scales1.shape[0]):
        weight_scales2_new.append(marlin_zero_points(weight_scales2[dim], n // GROUP_SIZE, k, 4))

    weight_scales2 = torch.stack(weight_scales2_new)
    del weight_scales2_new

    stream = torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with stream:
        for _ in range(1):
            marlin_output = torch.ops.vllm.fused_marlin_int2_moe(
                a,
                qweight1,
                qweight2,
                code_scales1,
                code_scales2,
                code_zeros1,
                code_zeros2,
                super_scales1,
                super_scales2,
                scores,
                topk_weights,
                topk_ids,
                global_num_experts=e,
                expert_map=None,
                w1_zeros=weight_scales1,
                w2_zeros=weight_scales2,
                quant_type_id=quant_type.id)

            torch.cuda.synchronize()
            diff = (fused_moe_out - marlin_output).abs()
            rdiff = diff / fused_moe_out.abs()
            torch.cuda.synchronize()
            eq = torch.allclose(fused_moe_out, marlin_output, rtol=1e-2, atol=1e-2)

    torch.cuda.synchronize()
    start_event.record(stream)
    torch.cuda.nvtx.range_push("marlin")
    with stream:
        for _ in range(0):
            marlin_output = torch.ops.vllm.fused_marlin_int2_moe(
                a,
                qweight1,
                qweight2,
                code_scales1,
                code_scales2,
                code_zeros1,
                code_zeros2,
                super_scales1,
                super_scales2,
                scores,
                topk_weights,
                topk_ids,
                global_num_experts=e,
                expert_map=None,
                w1_zeros=weight_scales1,
                w2_zeros=weight_scales2,
                quant_type_id=quant_type.id)
    end_event.record(stream)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Time taken: {elapsed_time} ms")
