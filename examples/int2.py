import torch
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

if __name__ == "__main__":
    GPTQ_MARLIN_TILE = 16
    GROUP_SIZE = 64

    device = torch.device("cuda:0")
    m = 666
    n = 1024
    k = 2048
    e = 12
    topk = 3
    dtype = torch.bfloat16
    quant_type = scalar_types.uint4

    a = torch.randn((m, k), dtype=dtype, device=device) / 10

    weight_storage_bits = 16
    weight_bits = 2
    weight_pack_factor = weight_storage_bits // weight_bits
    qweight1 = torch.zeros((e, k // GPTQ_MARLIN_TILE, 2 * n * GPTQ_MARLIN_TILE // weight_pack_factor),
                           dtype=torch.uint16,
                           device=device)
    qweight2 = torch.zeros((e, n // GPTQ_MARLIN_TILE, k * GPTQ_MARLIN_TILE // weight_pack_factor),
                           dtype=torch.uint16,
                           device=device)

    zp_storage_bits = 32
    zp_bits = 4
    zp_pack_factor = zp_storage_bits // zp_bits
    code_scales1 = torch.zeros((e, 2 * n), dtype=torch.float32, device=device)
    code_zeros1 = torch.zeros((e, 2 * n), dtype=torch.float32, device=device)
    zeros1 = torch.zeros((e, k // GROUP_SIZE, 2 * n // zp_pack_factor), dtype=torch.int32, device=device)
    super_scales1 = torch.zeros((e, 2 * n), dtype=dtype, device=device)

    code_scales2 = torch.zeros((e, k), dtype=torch.float32, device=device)
    code_zeros2 = torch.zeros((e, k), dtype=torch.float32, device=device)
    zeros2 = torch.zeros((e, n // GROUP_SIZE, k // zp_pack_factor), dtype=torch.int32, device=device)
    super_scales2 = torch.zeros((e, k), dtype=dtype, device=device)

    score = torch.randn((m, e), dtype=dtype, device=device)
    topk_weights, topk_ids, _ = fused_topk(a, score, topk, False)

    torch.cuda.nvtx.range_push("marlin")
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
        score,
        topk_weights,
        topk_ids,
        global_num_experts=e,
        expert_map=None,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        quant_type_id=quant_type.id)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
