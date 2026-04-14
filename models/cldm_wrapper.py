"""
ControlLDM - 一个完整的 ControlNet + LDM 封装类
参考了 ControlLDM 的设计模式，适配 diffusers 架构
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set, Optional, Union
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models import CLIPTextModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers import CLIPTokenizer

from .controlnet import ControlNetModel


class ControlLDM(nn.Module):
    """
    一个完整的 ControlNet + Latent Diffusion Model 封装类
    
    这个类整合了所有组件，提供了统一的接口和更好的权重管理。
    参考了 ControlLDM 的设计模式。
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        controlnet: ControlNetModel,
        latent_scale_factor: float = 0.18215,  # SD 的默认 scale factor
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.controlnet = controlnet
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13  # 默认 13 个控制尺度
        
    def freeze_base_models(self):
        """冻结基础模型（VAE、Text Encoder、UNet），只训练 ControlNet"""
        for module in [self.vae, self.text_encoder, self.unet]:
            module.eval()
            module.train = lambda mode: None  # 禁用训练模式
            for p in module.parameters():
                p.requires_grad = False
    
    def unfreeze_controlnet(self):
        """解冻 ControlNet 用于训练"""
        self.controlnet.train()
        for p in self.controlnet.parameters():
            p.requires_grad = True
    
    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str], Set[str]]:
        """
        从 UNet 初始化 ControlNet 权重
        
        Returns:
            init_with_new_zero: 需要新零初始化的键
            init_with_scratch: 需要从零开始的键
        """
        unet_sd = self.unet.state_dict()
        controlnet_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        
        for key in controlnet_sd:
            if key in unet_sd:
                this, target = controlnet_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    # 处理维度不匹配的情况（例如 conditioning embedding）
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype, device=target.device)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                # ControlNet 特有的层（如 zero conv）从零开始
                init_sd[key] = controlnet_sd[key].clone()
                init_with_scratch.add(key)
        
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(
        self,
        image: torch.Tensor,
        sample: bool = True,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        """
        VAE 编码图像到潜在空间
        
        Args:
            image: 输入图像，范围 [0, 1] 或 [-1, 1]
            sample: 是否采样（True）或使用 mode（False）
            tiled: 是否使用 tiled 编码（用于大图像）
            tile_size: tile 大小
            
        Returns:
            潜在空间表示
        """
        # 确保图像在正确范围
        if image.max() > 1.0:
            image = image / 255.0
        
        # 转换为 VAE 输入格式 [-1, 1]
        if image.min() >= 0:
            image = image * 2.0 - 1.0
        
        if tiled and tile_size > 0:
            # 使用 tiled VAE（需要 VAEHook 实现）
            # 这里简化处理，实际应该使用 VAEHook
            print(f"[ControlLDM] Using tiled VAE encoding with tile_size={tile_size}")
            # TODO: 集成 VAEHook 用于 tiled 编码
        
        with torch.no_grad():
            if sample:
                z = self.vae.encode(image).latent_dist.sample() * self.scale_factor
            else:
                z = self.vae.encode(image).latent_dist.mode() * self.scale_factor
        
        return z
    
    def vae_decode(
        self,
        z: torch.Tensor,
        tiled: bool = False,
        tile_size: int = -1,
    ) -> torch.Tensor:
        """
        VAE 解码潜在空间到图像
        
        Args:
            z: 潜在空间表示
            tiled: 是否使用 tiled 解码
            tile_size: tile 大小
            
        Returns:
            解码后的图像，范围 [-1, 1]
        """
        if tiled and tile_size > 0:
            print(f"[ControlLDM] Using tiled VAE decoding with tile_size={tile_size}")
            # TODO: 集成 VAEHook 用于 tiled 解码
        
        with torch.no_grad():
            image = self.vae.decode(z / self.scale_factor).sample
        
        return image
    
    def prepare_condition(
        self,
        cond_img: torch.Tensor,
        txt: Union[str, List[str]],
        tiled: bool = False,
        tile_size: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        准备条件输入（文本和图像）
        
        Args:
            cond_img: 条件图像
            txt: 文本提示
            tiled: 是否使用 tiled VAE
            tile_size: tile 大小
            
        Returns:
            包含文本和图像条件的字典
        """
        # 文本编码
        if isinstance(txt, str):
            txt = [txt]
        
        text_inputs = self.tokenizer(
            txt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.text_encoder.device))[0]
        
        # 图像编码到潜在空间
        # 确保图像在正确范围 [0, 1]
        if cond_img.max() > 1.0:
            cond_img = cond_img / 255.0
        
        c_img = self.vae_encode(
            cond_img * 2 - 1,  # 转换为 [-1, 1]
            sample=False,
            tiled=tiled,
            tile_size=tile_size,
        )
        
        return {
            "c_txt": text_embeddings,
            "c_img": c_img,
        }
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        control_scales: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_noisy: 噪声潜在空间
            t: 时间步
            cond: 条件字典，包含 'c_txt' 和 'c_img'
            control_scales: 控制尺度列表，如果为 None 则使用 self.control_scales
            
        Returns:
            预测的噪声
        """
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        
        # ControlNet 前向传播
        controlnet_output = self.controlnet(
            sample=x_noisy,
            timestep=t,
            encoder_hidden_states=c_txt,
            controlnet_cond=c_img,
            return_dict=False,
        )
        
        down_block_res_samples, mid_block_res_sample = controlnet_output
        
        # 应用控制尺度
        scales = control_scales if control_scales is not None else self.control_scales
        if len(scales) == len(down_block_res_samples) + 1:
            down_block_res_samples = [
                sample * scale for sample, scale in zip(down_block_res_samples, scales[:-1])
            ]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        else:
            # 如果尺度数量不匹配，使用统一的 conditioning_scale
            conditioning_scale = scales[0] if len(scales) > 0 else 1.0
            down_block_res_samples = [
                sample * conditioning_scale for sample in down_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale
        
        # UNet 前向传播
        eps = self.unet(
            x_noisy,
            timestep=t,
            encoder_hidden_states=c_txt,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        return eps
    
    def set_control_scales(self, scales: List[float]):
        """设置控制尺度"""
        if len(scales) != len(self.control_scales):
            print(f"[ControlLDM] Warning: scales length {len(scales)} != expected {len(self.control_scales)}")
        self.control_scales = scales
    
    def cast_dtype(self, dtype: torch.dtype):
        """转换模型到指定数据类型"""
        self.unet.to(dtype)
        self.controlnet.to(dtype)
        # VAE 和 Text Encoder 通常保持 float32 以获得更好的精度
        # self.vae.to(dtype)
        # self.text_encoder.to(dtype)
    
    def enable_xformers(self):
        """启用 xformers 内存高效注意力"""
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[ControlLDM] Warning: Failed to enable xformers: {e}")
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点以节省内存"""
        self.unet.enable_gradient_checkpointing()
        self.controlnet.enable_gradient_checkpointing()






