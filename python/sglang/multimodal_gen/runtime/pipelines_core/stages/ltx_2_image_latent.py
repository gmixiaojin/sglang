# Copied and adapted from native LTX-2 pipelines.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LTX2ImageLatentPreparationStage(PipelineStage):
    """Prepare packed image-latent tokens for LTX-2 TI2V (first-frame conditioning)."""

    def __init__(self, vae, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vae = vae

    @staticmethod
    def _resize_center_crop(
        img: PIL.Image.Image, *, width: int, height: int
    ) -> PIL.Image.Image:
        # Resize-to-fill then center-crop (native ltx-pipelines resize_and_center_crop)
        iw, ih = img.size
        scale = max(width / float(iw), height / float(ih))
        rw = int(round(iw * scale))
        rh = int(round(ih * scale))
        img = img.resize((rw, rh), resample=PIL.Image.Resampling.LANCZOS)
        left = (rw - width) // 2
        top = (rh - height) // 2
        img = img.crop((left, top, left + width, top + height))
        return img

    @staticmethod
    def _pil_to_normed_tensor(img: PIL.Image.Image) -> torch.Tensor:
        # PIL -> numpy [0,1] -> torch [B,C,H,W], then [-1,1]
        arr = pil_to_numpy(img)
        t = numpy_to_pt(arr)
        return normalize(t)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.image_latent is not None and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0:
            return batch

        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if batch.image_path is None:
            return batch
        if batch.width is None or batch.height is None:
            raise ValueError("width/height must be provided for LTX-2 TI2V.")

        image_path = (
            batch.image_path[0]
            if isinstance(batch.image_path, list)
            else batch.image_path
        )
        img = load_image(image_path)
        img = self._resize_center_crop(img, width=int(batch.width), height=int(batch.height))
        batch.condition_image = img

        image_tensor = self._pil_to_normed_tensor(img).to(get_local_torch_device(), dtype=torch.float32)
        # [B, C, H, W] -> [B, C, 1, H, W]
        video_condition = image_tensor.unsqueeze(2)

        self.vae = self.vae.to(get_local_torch_device())

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)

            latent_dist: DiagonalGaussianDistribution = self.vae.encode(video_condition)
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if mode == "argmax":
            latent = latent_dist.mode()
        elif mode == "sample":
            if batch.generator is None:
                raise ValueError("Generator must be provided for VAE sampling.")
            latent = latent_dist.sample(batch.generator)
        else:
            raise ValueError(f"Unsupported encode_sample_mode: {mode}")

        latent = server_args.pipeline_config.postprocess_vae_encode(latent, self.vae)

        scaling_factor, shift_factor = server_args.pipeline_config.get_decode_scale_and_shift(
            device=latent.device, dtype=latent.dtype, vae=self.vae
        )
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent.device)
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent.device)
        latent = (latent - shift_factor) * scaling_factor

        packed = server_args.pipeline_config.maybe_pack_latents(latent, latent.shape[0], batch)
        if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
            raise ValueError("Expected packed image latents [B, S0, D].")

        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        latent_h = int(batch.height) // vae_sf
        latent_w = int(batch.width) // vae_sf
        expected_tokens = (latent_h // patch) * (latent_w // patch)
        if int(packed.shape[1]) != int(expected_tokens):
            raise ValueError(
                "LTX-2 conditioning token count mismatch: "
                f"{int(packed.shape[1])=} {int(expected_tokens)=}."
            )

        batch.image_latent = packed
        batch.ltx2_num_image_tokens = int(packed.shape[1])

        if batch.debug:
            logger.info(
                "LTX2 image conditioning prepared: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(batch.image_latent.shape),
                batch.width,
                batch.height,
            )

        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

