"""Example usage for the ControlLDM wrapper."""

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models import CLIPTextModel
from transformers import CLIPTokenizer

from models.cldm_wrapper import ControlLDM
from models.controlnet import ControlNetModel


def create_controllm(
    pretrained_model_path: str,
    daspsr_model_path: str = None,
) -> ControlLDM:
    """Create a ControlLDM instance."""
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")

    if daspsr_model_path:
        unet = UNet2DConditionModel.from_pretrained(daspsr_model_path, subfolder="unet")
        controlnet = ControlNetModel.from_pretrained(daspsr_model_path, subfolder="controlnet")
    else:
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
        controlnet = ControlNetModel.from_unet(unet, use_image_cross_attention=True)

    return ControlLDM(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
    )


def example_training_setup():
    """Example training setup."""
    controllm = create_controllm("path/to/pretrained_model")
    controllm.freeze_base_models()
    controllm.unfreeze_controlnet()
    controllm.enable_gradient_checkpointing()
    controllm.enable_xformers()
    controllm.set_control_scales([1.0] * 13)
    return controllm


def example_inference():
    """Example inference setup."""
    controllm = create_controllm("path/to/pretrained_model", "path/to/daspsr_model")
    controllm.eval()
    controllm.cast_dtype(torch.float16)

    cond_img = torch.randn(1, 3, 512, 512)
    prompt = "a beautiful landscape"
    cond = controllm.prepare_condition(
        cond_img=cond_img,
        txt=prompt,
        tiled=True,
        tile_size=256,
    )

    DDPMScheduler.from_pretrained("path/to/pretrained_model", subfolder="scheduler")
    latents = torch.randn(1, 4, 64, 64)
    timesteps = torch.tensor([500])

    with torch.no_grad():
        noise_pred = controllm(
            x_noisy=latents,
            t=timesteps,
            cond=cond,
        )

    return noise_pred


def example_from_unet_init():
    """Example ControlNet initialization from UNet weights."""
    controllm = create_controllm("path/to/pretrained_model")
    init_with_new_zero, init_with_scratch = controllm.load_controlnet_from_unet()

    print(f"Keys initialized with new zeros: {len(init_with_new_zero)}")
    print(f"Keys initialized from scratch: {len(init_with_scratch)}")

    return controllm


if __name__ == "__main__":
    print("=== Training Setup ===")
    train_model = example_training_setup()

    print("\n=== Inference Example ===")
    # noise_pred = example_inference()

    print("\n=== From UNet Init ===")
    # init_model = example_from_unet_init()
