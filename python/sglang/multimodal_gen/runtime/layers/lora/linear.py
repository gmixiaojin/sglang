# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Code adapted from SGLang https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/layers.py

import math

import torch
from torch import nn
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_tp_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import get_mixed_precision_state

logger = init_logger(__name__)

torch._dynamo.config.recompile_limit = 16


class BaseLayerWithLoRA(nn.Module):

    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer

        self.merged: bool = False
        self.cpu_weight = base_layer.weight.to("cpu")
        # indicates adapter weights don't contain this layer
        # (which shouldn't normally happen, but we want to separate it from the case of erroneous merging)
        self.disable_lora: bool = False
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.training_mode = training_mode
        self.lora_path: str | None = None

        # Multi-LoRA batching support (merge-based only)
        self.use_multi_lora: bool = False
        self.active_lora_indices: torch.Tensor | None = None
        self.lora_weights_pool: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.lora_nickname_to_index: dict[str, int] = {}
        self.lora_adapter_configs: dict[str, dict] = {}
        self.layer_name: str | None = None
        self.merged_weights_pool: dict[str, torch.Tensor] = {}  # Pre-computed merged weights

        if training_mode:
            assert (
                self.lora_rank is not None
            ), "LoRA rank  must be set for training mode"
            if self.lora_rank is None or self.lora_alpha is None:
                self.lora_alpha = lora_rank
            self.base_layer.requires_grad_(False)
            in_dim = self.base_layer.weight.shape[1]
            out_dim = self.base_layer.weight.shape[0]
            self.lora_A = nn.Parameter(
                torch.zeros(
                    self.lora_rank,
                    in_dim,
                    device=self.base_layer.weight.device,
                    dtype=self.base_layer.weight.dtype,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    out_dim,
                    self.lora_rank,
                    device=self.base_layer.weight.device,
                    dtype=self.base_layer.weight.dtype,
                )
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    @torch.compile()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_A = self.lora_A
        lora_B = self.lora_B
        if isinstance(self.lora_B, DTensor):
            lora_B = self.lora_B.to_local()
            lora_A = self.lora_A.to_local()

        if not self.merged and not self.disable_lora:
            lora_A_sliced = self.slice_lora_a_weights(lora_A.to(x, non_blocking=True))
            lora_B_sliced = self.slice_lora_b_weights(lora_B.to(x, non_blocking=True))
            delta = x @ lora_A_sliced.T @ lora_B_sliced.T
            if self.lora_alpha != self.lora_rank:
                delta = delta * (
                    self.lora_alpha / self.lora_rank  # type: ignore
                )  # type: ignore
            out, output_bias = self.base_layer(x)
            return out + delta, output_bias
        else:
            out, output_bias = self.base_layer(x)
            return out.to(x), output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B

    def _index_to_nickname(self, lora_idx: int) -> str | None:
        """Convert LoRA index to nickname using reverse lookup."""
        for nickname, idx in self.lora_nickname_to_index.items():
            if idx == lora_idx:
                return nickname
        return None

    def set_multi_lora_state(
        self,
        active_lora_indices: torch.Tensor,
        lora_weights_pool: dict[str, tuple[torch.Tensor, torch.Tensor]],
        lora_nickname_to_index: dict[str, int],
        lora_adapter_configs: dict[str, dict],
    ) -> None:
        """Set the multi-LoRA state for this layer and pre-merge weights."""
        self.use_multi_lora = True
        self.active_lora_indices = active_lora_indices
        self.lora_weights_pool = lora_weights_pool
        self.lora_nickname_to_index = lora_nickname_to_index
        self.lora_adapter_configs = lora_adapter_configs
        
        # Always pre-merge weights for fully-fused forward
        if not self.merged_weights_pool:
            self._premerge_lora_weights_pool()

    def clear_multi_lora_state(self) -> None:
        """Clear the multi-LoRA state."""
        self.use_multi_lora = False
        self.active_lora_indices = None
        self.lora_weights_pool = {}
        self.lora_nickname_to_index = {}
        self.lora_adapter_configs = {}
        # Note: merged_weights_pool is kept for reuse (can be cleared explicitly if needed)

    @torch.no_grad()
    def _premerge_lora_weights_pool(self) -> None:
        """Pre-merge all LoRA weights in the pool to enable fully-fused forward.
        
        This computes W' = W_base + B @ A for each LoRA in the pool.
        The merged weights are stored for efficient batched forward.
        Note: For TP, merged weights are sharded according to the layer type.
        """
        if not self.lora_weights_pool:
            return
        
        device = get_local_torch_device()
        base_weight = self.base_layer.weight
        
        # Get base weight (handle DTensor and TP sharding)
        if isinstance(base_weight, DTensor):
            base_weight_data = base_weight.to_local().to(device)
        else:
            base_weight_data = base_weight.to(device)
        
        self.merged_weights_pool = {}
        
        for nickname, (lora_A, lora_B) in self.lora_weights_pool.items():
            lora_A = lora_A.to(device, non_blocking=True)
            lora_B = lora_B.to(device, non_blocking=True)
            
            # Apply slicing for TP (subclasses override these methods)
            lora_A_sliced = self.slice_lora_a_weights(lora_A)
            lora_B_sliced = self.slice_lora_b_weights(lora_B)
            
            # Compute delta: delta = B @ A (with alpha scaling)
            delta = lora_B_sliced @ lora_A_sliced
            
            # Apply alpha scaling if needed
            adapter_config = self.lora_adapter_configs.get(nickname, {})
            alpha = adapter_config.get("alpha", self.lora_alpha or 16.0)
            rank = adapter_config.get("rank", self.lora_rank or 16)
            if alpha != rank:
                delta = delta * (alpha / rank)
            
            # Compute merged weight: W' = W_base + delta
            # Note: base_weight_data is already sharded for TP layers
            merged_weight = base_weight_data.clone()
            merged_weight += delta
            
            self.merged_weights_pool[nickname] = merged_weight
        
        logger.debug(
            "Pre-merged %d LoRA weights for layer %s (TP-aware)",
            len(self.merged_weights_pool),
            self.layer_name or self.__class__.__name__,
        )

    def set_lora_weights(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        training_mode: bool = False,
        lora_path: str | None = None,
    ) -> None:
        self.lora_A = torch.nn.Parameter(
            A
        )  # share storage with weights in the pipeline
        self.lora_B = torch.nn.Parameter(B)
        self.disable_lora = False
        if not training_mode:
            self.merge_lora_weights()
        self.lora_path = lora_path

    @torch.no_grad()
    def merge_lora_weights(self) -> None:
        if self.disable_lora:
            return

        if self.merged:
            self.unmerge_lora_weights()
        assert (
            self.lora_A is not None and self.lora_B is not None
        ), "LoRA weights not set. Please set them first."
        if isinstance(self.base_layer.weight, DTensor):
            mesh = self.base_layer.weight.data.device_mesh
            unsharded_base_layer = ReplicatedLinear(
                input_size=self.base_layer.input_size,
                output_size=self.base_layer.output_size,
                bias=getattr(self.base_layer, "bias", None) is not None,
                skip_bias_add=self.base_layer.skip_bias_add,
                params_dtype=self.base_layer.params_dtype,
                quant_config=self.base_layer.quant_config,
                prefix=self.base_layer.prefix,
            )
            # Using offload param is on CPU, so current_device is for "CPU -> GPU -> merge -> CPU"
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(
                get_local_torch_device()
            ).full_tensor()
            data += self.slice_lora_b_weights(self.lora_B).to(
                data
            ) @ self.slice_lora_a_weights(self.lora_A).to(data)
            unsharded_base_layer.weight = nn.Parameter(data.to(current_device))
            if isinstance(getattr(self.base_layer, "bias", None), DTensor):
                unsharded_base_layer.bias = nn.Parameter(
                    self.base_layer.bias.to(get_local_torch_device(), non_blocking=True)
                    .full_tensor()
                    .to(current_device)
                )

            offload_policy = (
                CPUOffloadPolicy() if "cpu" in str(current_device) else OffloadPolicy()
            )
            mp_policy = get_mixed_precision_state().mp_policy

            self.base_layer = fully_shard(
                unsharded_base_layer,
                mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
            )
        else:
            current_device = self.base_layer.weight.data.device
            data = self.base_layer.weight.data.to(get_local_torch_device())
            data += self.slice_lora_b_weights(
                self.lora_B.to(data)
            ) @ self.slice_lora_a_weights(self.lora_A.to(data))
            self.base_layer.weight.data = data.to(current_device, non_blocking=True)

        self.merged = True

    @torch.no_grad()
    # @torch.compile(dynamic=True)
    def unmerge_lora_weights(self) -> None:
        if self.disable_lora:
            return

        if not self.merged:
            raise ValueError(
                "LoRA weights not merged. Please merge them first before unmerging."
            )

        # avoid precision loss
        if isinstance(self.base_layer.weight, DTensor):
            device = self.base_layer.weight.data.device
            self.base_layer.weight = nn.Parameter(
                self.cpu_weight.to(device, non_blocking=True)
            )
        else:
            self.base_layer.weight.data = self.cpu_weight.data.to(
                self.base_layer.weight, non_blocking=True
            )

        self.merged = False


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
    ) -> None:
        super().__init__(base_layer)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "We don't support VocabParallelEmbeddingWithLoRA yet."
        )


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def _forward_with_merged_weights(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward using pre-merged weights pool (fully fuse mode) for ColumnParallel."""
        batch_size = input_.shape[0]
        device = input_.device
        
        # Get base weight for fallback
        if isinstance(self.base_layer.weight, DTensor):
            base_weight = self.base_layer.weight.to_local().to(device)
        else:
            base_weight = self.base_layer.weight.to(device)
        
        # Gather merged weights for each sample
        selected_weights = []
        for i in range(batch_size):
            lora_idx = self.active_lora_indices[i].item()
            if lora_idx < 0:
                selected_weights.append(base_weight)
            else:
                nickname = self._index_to_nickname(lora_idx)
                if nickname and nickname in self.merged_weights_pool:
                    selected_weights.append(self.merged_weights_pool[nickname].to(device))
                else:
                    selected_weights.append(base_weight)
        
        weights_stack = torch.stack(selected_weights, dim=0)
        
        # Batched matmul
        if input_.dim() == 2:
            input_expanded = input_.unsqueeze(1)
            output_parallel = torch.bmm(input_expanded, weights_stack.transpose(-1, -2)).squeeze(1)
        elif input_.dim() == 3:
            output_parallel = torch.bmm(input_, weights_stack.transpose(-1, -2))
        else:
            raise ValueError(f"Unsupported input dimension: {input_.dim()}")
        
        return output_parallel

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # Handle multi-LoRA with merged weights (fully fuse mode)
        if (
            self.use_multi_lora
            and self.active_lora_indices is not None
            and not self.disable_lora
            and self.merged_weights_pool
        ):
            output_parallel = self._forward_with_merged_weights(input_)
        else:
            # Standard forward (no multi-LoRA)
            bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_, bias
            )
        
        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A.to(self.base_layer.weight)

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):

    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        return A

    def slice_lora_b_weights(
        self, B: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tp_rank = get_tp_rank()
        B_q, B_kv = B
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    """
    Row parallel linear layer with LoRA support.

    In tensor parallelism, RowParallel splits the input dimension.
    For LoRA: A is sliced along input dim, B is full.
    The LoRA delta must be all-reduced across TP ranks.
    """

    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        training_mode: bool = False,
    ) -> None:
        super().__init__(base_layer, lora_rank, lora_alpha, training_mode)

    def _forward_with_merged_weights(self, input_parallel: torch.Tensor) -> torch.Tensor:
        """Forward using pre-merged weights pool (fully fuse mode).
        
        This uses batched matmul with selected weights for optimal performance.
        Supports both 2D (batch, in_dim) and 3D (batch, seq_len, in_dim) inputs.
        """
        batch_size = input_parallel.shape[0]
        device = input_parallel.device
        
        # Get base weight for fallback
        if isinstance(self.base_layer.weight, DTensor):
            base_weight = self.base_layer.weight.to_local().to(device)
        else:
            base_weight = self.base_layer.weight.to(device)
        
        # Gather merged weights for each sample
        selected_weights = []
        for i in range(batch_size):
            lora_idx = self.active_lora_indices[i].item()
            if lora_idx < 0:
                # Use base weight
                selected_weights.append(base_weight)
            else:
                nickname = self._index_to_nickname(lora_idx)
                if nickname and nickname in self.merged_weights_pool:
                    selected_weights.append(self.merged_weights_pool[nickname].to(device))
                else:
                    # Fallback to base weight
                    selected_weights.append(base_weight)
        
        # Stack weights: (batch_size, out_dim, in_dim)
        weights_stack = torch.stack(selected_weights, dim=0)
        
        # Batched matmul
        if input_parallel.dim() == 2:
            # (batch, in_dim) -> (batch, 1, in_dim)
            input_expanded = input_parallel.unsqueeze(1)
            # (batch, 1, in_dim) @ (batch, in_dim, out_dim) -> (batch, 1, out_dim)
            output_parallel = torch.bmm(input_expanded, weights_stack.transpose(-1, -2)).squeeze(1)
        elif input_parallel.dim() == 3:
            # (batch, seq_len, in_dim) @ (batch, in_dim, out_dim) -> (batch, seq_len, out_dim)
            output_parallel = torch.bmm(input_parallel, weights_stack.transpose(-1, -2))
        else:
            raise ValueError(f"Unsupported input dimension: {input_parallel.dim()}")
        
        return output_parallel

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tp_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()

        # Handle multi-LoRA with merged weights (fully fuse mode)
        if (
            self.use_multi_lora
            and self.active_lora_indices is not None
            and not self.disable_lora
            and self.merged_weights_pool
        ):
            output_parallel = self._forward_with_merged_weights(input_parallel)
            output_bias = (
                self.base_layer.bias if self.base_layer.skip_bias_add else None
            )
        else:
            # Standard forward (no multi-LoRA)
            output_parallel = self.base_layer.quant_method.apply(
                self.base_layer, input_parallel
            )
            output_bias = (
                self.base_layer.bias if self.base_layer.skip_bias_add else None
            )

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    def slice_lora_a_weights(self, A: torch.Tensor) -> torch.Tensor:
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor) -> torch.Tensor:
        return B


def get_lora_layer(
    layer: nn.Module,
    lora_rank: int | None = None,
    lora_alpha: int | None = None,
    training_mode: bool = False,
) -> BaseLayerWithLoRA | None:
    supported_layer_types: dict[type[LinearBase], type[BaseLayerWithLoRA]] = {
        # the order matters
        # VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
        ReplicatedLinear: BaseLayerWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(
                layer,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                training_mode=training_mode,
            )
            return ret
    return None


# source: https://github.com/vllm-project/vllm/blob/93b38bea5dd03e1b140ca997dfaadef86f8f1855/vllm/lora/utils.py#L9
def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module:
    """Replace a submodule in a model with a new module."""
    parent = model.get_submodule(".".join(module_name.split(".")[:-1]))
    target_name = module_name.split(".")[-1]
    setattr(parent, target_name, new_module)
    return new_module
