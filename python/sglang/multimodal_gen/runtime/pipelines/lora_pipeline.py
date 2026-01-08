# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    get_lora_layer,
    replace_submodule,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import OutputBatch, Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_lora
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LoRAPipeline(ComposedPipelineBase):
    # Pipeline that supports injecting LoRA adapters into the diffusion transformer.

    lora_adapters: dict[str, dict[str, torch.Tensor]] = defaultdict(
        dict
    )  # state dicts of loaded lora adapters
    cur_adapter_name: str = ""
    cur_adapter_path: str = ""
    lora_layers: dict[str, BaseLayerWithLoRA] = {}
    lora_layers_critic: dict[str, BaseLayerWithLoRA] = {}
    server_args: ServerArgs
    exclude_lora_layers: list[str] = []
    device: torch.device = get_local_torch_device()
    lora_target_modules: list[str] | None = None
    lora_path: str | None = None
    lora_nickname: str = "default"
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_initialized: bool = False

    # Multi-LoRA batching support
    _lora_nickname_to_index: dict[str, int] = {}
    _lora_adapter_configs: dict[str, dict] = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = get_local_torch_device()
        self.exclude_lora_layers = self.modules[
            "transformer"
        ].config.arch_config.exclude_lora_layers
        self.lora_target_modules = self.server_args.lora_target_modules
        self.lora_path = self.server_args.lora_path
        self.lora_nickname = self.server_args.lora_nickname
        if self.lora_path is not None:
            self.convert_to_lora_layers()
            self.set_lora_adapter(
                self.lora_nickname, self.lora_path  # type: ignore
            )  # type: ignore

    def is_target_layer(self, module_name: str) -> bool:
        if self.lora_target_modules is None:
            return True
        return any(
            target_name in module_name for target_name in self.lora_target_modules
        )

    def convert_to_lora_layers(self) -> None:
        # Unified method to convert the transformer to a LoRA transformer.
        if self.lora_initialized:
            return
        self.lora_initialized = True
        converted_count = 0
        for name, layer in self.modules["transformer"].named_modules():
            if not self.is_target_layer(name):
                continue

            excluded = False
            for exclude_layer in self.exclude_lora_layers:
                if exclude_layer in name:
                    excluded = True
                    break
            if excluded:
                continue

            layer = get_lora_layer(
                layer,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            if layer is not None:
                self.lora_layers[name] = layer
                replace_submodule(self.modules["transformer"], name, layer)
                converted_count += 1
        logger.info("Converted %d layers to LoRA layers", converted_count)

        if "fake_score_transformer" in self.modules:
            for name, layer in self.modules["fake_score_transformer"].named_modules():
                if not self.is_target_layer(name):
                    continue
                layer = get_lora_layer(
                    layer,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                )
                if layer is not None:
                    self.lora_layers_critic[name] = layer
                    replace_submodule(
                        self.modules["fake_score_transformer"], name, layer
                    )
                    converted_count += 1
            logger.info(
                "Converted %d layers to LoRA layers in the critic model",
                converted_count,
            )

    def set_lora_adapter(
        self, lora_nickname: str, lora_path: str | None = None
    ):  
        # Load a LoRA adapter into the pipeline and merge it into the transformer.

        if lora_nickname not in self.lora_adapters and lora_path is None:
            raise ValueError(
                f"Adapter {lora_nickname} not found in the pipeline. Please provide lora_path to load it."
            )
        if not self.lora_initialized:
            self.convert_to_lora_layers()
        adapter_updated = False
        rank = dist.get_rank()
        if lora_path is not None and lora_path != self.cur_adapter_path:
            lora_local_path = maybe_download_lora(lora_path)
            lora_state_dict = load_file(lora_local_path)

            # Map the hf layer names to our custom layer names
            param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].param_names_mapping
            )
            lora_param_names_mapping_fn = get_param_names_mapping(
                self.modules["transformer"].lora_param_names_mapping
            )

            to_merge_params: defaultdict[Hashable, dict[Any, Any]] = defaultdict(dict)
            for name, weight in lora_state_dict.items():
                name = name.replace("diffusion_model.", "")
                name = name.replace(".weight", "")
                name, _, _ = lora_param_names_mapping_fn(name)
                target_name, merge_index, num_params_to_merge = param_names_mapping_fn(
                    name
                )
                # for (in_dim, r) @ (r, out_dim), we only merge (r, out_dim * n) where n is the number of linear layers to fuse
                # see param mapping in HunyuanVideoArchConfig
                if merge_index is not None and "lora_B" in name:
                    to_merge_params[target_name][merge_index] = weight
                    if len(to_merge_params[target_name]) == num_params_to_merge:
                        # cat at output dim according to the merge_index order
                        sorted_tensors = [
                            to_merge_params[target_name][i]
                            for i in range(num_params_to_merge)
                        ]
                        weight = torch.cat(sorted_tensors, dim=1)
                        del to_merge_params[target_name]
                    else:
                        continue

                if target_name in self.lora_adapters[lora_nickname]:
                    raise ValueError(
                        f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                    )
                self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
            adapter_updated = True
            self.cur_adapter_path = lora_path
            logger.info("Rank %d: loaded LoRA adapter %s", rank, lora_path)

        if not adapter_updated and self.cur_adapter_name == lora_nickname:
            return
        self.cur_adapter_name = lora_nickname

        # Merge the new adapter
        adapted_count = 0
        for name, layer in self.lora_layers.items():
            lora_A_name = name + ".lora_A"
            lora_B_name = name + ".lora_B"
            if (
                lora_A_name in self.lora_adapters[lora_nickname]
                and lora_B_name in self.lora_adapters[lora_nickname]
            ):
                layer.set_lora_weights(
                    self.lora_adapters[lora_nickname][lora_A_name],
                    self.lora_adapters[lora_nickname][lora_B_name],
                    lora_path=lora_path,
                )
                adapted_count += 1
            else:
                if rank == 0:
                    logger.warning(
                        "LoRA adapter %s does not contain the weights for layer %s. LoRA will not be applied to it.",
                        lora_path,
                        name,
                    )
                layer.disable_lora = True
        logger.info(
            "Rank %d: LoRA adapter %s applied to %d layers",
            rank,
            lora_path,
            adapted_count,
        )

    def merge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.merge_lora_weights()

    def unmerge_lora_weights(self) -> None:
        for name, layer in self.lora_layers.items():
            layer.unmerge_lora_weights()

    def _load_lora_adapter(self, lora_path: str, lora_nickname: str) -> None:
        lora_local_path = maybe_download_lora(lora_path)
        lora_state_dict = load_file(lora_local_path)

        # Map the hf layer names to our custom layer names
        param_names_mapping_fn = get_param_names_mapping(
            self.modules["transformer"].param_names_mapping
        )
        lora_param_names_mapping_fn = get_param_names_mapping(
            self.modules["transformer"].lora_param_names_mapping
        )

        to_merge_params: defaultdict[Hashable, dict[Any, Any]] = defaultdict(dict)
        for name, weight in lora_state_dict.items():
            name = name.replace("diffusion_model.", "")
            name = name.replace(".weight", "")
            name, _, _ = lora_param_names_mapping_fn(name)
            target_name, merge_index, num_params_to_merge = param_names_mapping_fn(name)
            # for (in_dim, r) @ (r, out_dim), we only merge (r, out_dim * n) where n is the number of linear layers to fuse
            # see param mapping in HunyuanVideoArchConfig
            if merge_index is not None and "lora_B" in name:
                to_merge_params[target_name][merge_index] = weight
                if len(to_merge_params[target_name]) == num_params_to_merge:
                    # cat at output dim according to the merge_index order
                    sorted_tensors = [
                        to_merge_params[target_name][i]
                        for i in range(num_params_to_merge)
                    ]
                    weight = torch.cat(sorted_tensors, dim=1)
                    del to_merge_params[target_name]
                else:
                    continue

            if target_name in self.lora_adapters[lora_nickname]:
                raise ValueError(
                    f"Target name {target_name} already exists in lora_adapters[{lora_nickname}]"
                )
            self.lora_adapters[lora_nickname][target_name] = weight.to(self.device)
        logger.info("Rank %d: loaded LoRA adapter %s", dist.get_rank(), lora_path)

    def preload_lora_adapter(
        self,
        lora_nickname: str,
        lora_path: str,
        alpha: float | None = None,
        rank: int | None = None,
    ) -> None:

        if not self.lora_initialized:
            self.convert_to_lora_layers()

        # Load adapter if not already loaded
        if lora_nickname not in self.lora_adapters:
            self._load_lora_adapter(lora_path, lora_nickname)

        # Assign index
        if lora_nickname not in self._lora_nickname_to_index:
            self._lora_nickname_to_index[lora_nickname] = len(
                self._lora_nickname_to_index
            )

        # Infer rank from weights if not provided
        if rank is None:
            for weight_name, weight in self.lora_adapters[lora_nickname].items():
                if "lora_A" in weight_name:
                    rank = weight.shape[0]
                    break
            rank = rank or 16

        self._lora_adapter_configs[lora_nickname] = {
            "alpha": alpha if alpha is not None else rank,
            "rank": rank,
        }
        logger.info(
            "Preloaded LoRA '%s' (index=%d)",
            lora_nickname,
            self._lora_nickname_to_index[lora_nickname],
        )

    def _set_multi_lora_state_on_layers(
        self, batch: Req, target_layers: dict[str, BaseLayerWithLoRA]
    ) -> None:
        # Set multi-LoRA state for on-the-fly computation.

        # Build active_lora_indices per sample
        batch_size = batch.batch_size
        
        # Support per-sample LoRA
        if isinstance(batch.lora_nickname, list):
            # Per-sample LoRA: [nickname1, nickname2, ...]
            active_indices = torch.zeros(
                (batch_size,), dtype=torch.long, device=self.device
            )
            for i, nickname in enumerate(batch.lora_nickname):
                if nickname in self._lora_nickname_to_index:
                    active_indices[i] = self._lora_nickname_to_index[nickname]
                else:
                    active_indices[i] = -1  # No LoRA
        elif batch.lora_nickname is not None:
            # Single LoRA for all samples
            if batch.lora_nickname not in self._lora_nickname_to_index:
                logger.warning("LoRA '%s' not preloaded, skipping", batch.lora_nickname)
                return
            lora_idx = self._lora_nickname_to_index[batch.lora_nickname]
            active_indices = torch.full(
                (batch_size,), lora_idx, dtype=torch.long, device=self.device
            )
        else:
            return  # No LoRA requested
        
        # Set state on all layers
        for layer_name, layer in target_layers.items():
            # Build weights pool for this layer
            lora_weights_pool: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
            lora_A_name = layer_name + ".lora_A"
            lora_B_name = layer_name + ".lora_B"
            
            for nickname in self._lora_nickname_to_index.keys():
                if (
                    lora_A_name in self.lora_adapters[nickname]
                    and lora_B_name in self.lora_adapters[nickname]
                ):
                    lora_weights_pool[nickname] = (
                        self.lora_adapters[nickname][lora_A_name],
                        self.lora_adapters[nickname][lora_B_name],
                    )
            
            layer.set_multi_lora_state(
                active_lora_indices=active_indices,
                lora_weights_pool=lora_weights_pool,
                lora_nickname_to_index=self._lora_nickname_to_index,
                lora_adapter_configs=self._lora_adapter_configs,
            )
            layer.layer_name = layer_name

    def _clear_multi_lora_state_on_layers(
        self, target_layers: dict[str, BaseLayerWithLoRA]
    ) -> None:
        # Clear multi-LoRA state from layers.
        for layer in target_layers.values():
            layer.clear_multi_lora_state()

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        # Check if multi-LoRA is requested
        if isinstance(batch.lora_nickname, list):
            # Per-sample LoRA: check if any nickname is valid
            use_multi_lora = any(
                nickname in self._lora_nickname_to_index
                for nickname in batch.lora_nickname
            )
        elif batch.lora_nickname is not None:
            use_multi_lora = batch.lora_nickname in self._lora_nickname_to_index
        else:
            use_multi_lora = False

        if use_multi_lora:
            self._set_multi_lora_state_on_layers(batch, self.lora_layers)
            if self.lora_layers_critic:
                self._set_multi_lora_state_on_layers(batch, self.lora_layers_critic)

        try:
            result = super().forward(batch, server_args)
        finally:
            if use_multi_lora:
                self._clear_multi_lora_state_on_layers(self.lora_layers)
                if self.lora_layers_critic:
                    self._clear_multi_lora_state_on_layers(self.lora_layers_critic)

        return result
