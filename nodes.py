import os
import torch

import folder_paths
import yaml
import comfy.model_management as mm
from comfy.utils import ProgressBar, load_torch_file

from omegaconf import OmegaConf
from tqdm import tqdm
import cv2

from .gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from .gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F

from .gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from .gimmvfi.generalizable_INR.raft import RAFT
from .gimmvfi.generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from .gimmvfi.generalizable_INR.flowformer.configs.submission import get_cfg
from .gimmvfi.utils.flow_viz import flow_to_image
from .gimmvfi.utils.utils import InputPadder, RaftArgs, easydict_to_dict

from contextlib import nullcontext

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))


class DownloadAndLoadGIMMVFIModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                    "gimmvfi_r_arb_lpips_fp32.safetensors",
                    "gimmvfi_f_arb_lpips_fp32.safetensors"
                    ],),
               },
               "optional": {
                    "precision": (["fp32", "bf16", "fp16"], {"default": "fp32"}),
                    "torch_compile": ("BOOLEAN", {"default": False, "tooltip": "Compile part of the model with torch.compile, requires Triton"}),
               },
        }

    RETURN_TYPES = ("GIMMVIF_MODEL",)
    RETURN_NAMES = ("gimmvfi_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "GIMM-VFI"
    DESCRIPTION = "Downloads and loads GIMM-VFI model from folder 'ComfyUI\models\interpolation\gimm-vfi'"

    def loadmodel(self, model, precision="fp32", torch_compile=False):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[precision]

        download_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi')
        model_path = os.path.join(download_path, model)

        if not os.path.exists(model_path):
            log.info(f"Downloading GMMI-VFI model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        if "gimmvfi_r" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_r_arb.yaml")
            flow_model = "raft-things_fp32.safetensors"
        elif "gimmvfi_f" in model:
            config_path = os.path.join(script_directory, "configs", "gimmvfi", "gimmvfi_f_arb.yaml")
            flow_model = "flowformer_sintel_fp32.safetensors"

        flow_model_path = os.path.join(folder_paths.models_dir, 'interpolation', 'gimm-vfi', flow_model)

        if not os.path.exists(flow_model_path):
            log.info(f"Downloading RAFT model to: {flow_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Kijai/GIMM-VFI_safetensors",
                allow_patterns=[f"*{flow_model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )
       
            
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
        arch_defaults = GIMMVFIConfig.create(config.arch)
        config = OmegaConf.merge(arch_defaults, config.arch)

        # load model
        if "gimmvfi_r" in model:
            model = GIMMVFI_R(dtype, config)
             #load RAFT
            raft_args = RaftArgs(
                small=False,
                mixed_precision=False,
                alternate_corr=False
            )
        
            raft_model = RAFT(raft_args)
            raft_sd = load_torch_file(flow_model_path)
            raft_model.load_state_dict(raft_sd, strict=True)
            raft_model.to(dtype).to(device)
            flow_estimator = raft_model
        elif "gimmvfi_f" in model:
            model = GIMMVFI_F(dtype, config)
            cfg = get_cfg()
            flowformer = FlowFormer(cfg.latentcostformer)
            flowformer_sd = load_torch_file(flow_model_path)
            flowformer.load_state_dict(flowformer_sd, strict=True)
            flow_estimator = flowformer.to(dtype).to(device)
            
       
        sd = load_torch_file(model_path)
        model.load_state_dict(sd, strict=False)
      
        model.flow_estimator = flow_estimator
        model = model.eval().to(dtype).to(device)

        if torch_compile:
            model = torch.compile(model)
            
        return (model,)

#region Interpolate
class GIMMVFI_interpolate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gimmvfi_model": ("GIMMVIF_MODEL",),
                "images": ("IMAGE", {"tooltip": "The images to interpolate between"}),
                "ds_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "interpolation_factor": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1, 
                    "tooltip": "Number of frame pairs to process in parallel. Higher = faster on A100. Recommended: 8-16 for A100 80GB, 4-8 for RTX 4090."}),
                "output_flows": ("BOOLEAN", {"default": False, "tooltip": "Output the flow tensors"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("images", "flow_tensors",)
    FUNCTION = "interpolate"
    CATEGORY = "PyramidFlowWrapper"

    def interpolate(self, gimmvfi_model, images, ds_factor, interpolation_factor, seed, 
                    batch_size=8, output_flows=False):
        mm.soft_empty_cache()
        images = images.permute(0, 3, 1, 2)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = gimmvfi_model.dtype
        total_pairs = images.shape[0] - 1
        
        # Get image dimensions for batch size calculation
        _, _, H, W = images.shape
        pixels = H * W
        
        # Limit batch size based on resolution to avoid integer overflow
        # The correlation tensor has shape [batch * H/8 * W/8, 1, H/8, W/8]
        # Integer overflow occurs when batch * (H/8) * (W/8) > 2^31
        # Safe limit: batch * pixels/64 < 2^31 -> batch < 2^31 * 64 / pixels
        max_batch_for_resolution = max(1, int((2**30) / (pixels // 64 + 1)))
        
        # Adjust batch_size based on available VRAM (auto-scale for safety)
        if torch.cuda.is_available():
            free_vram = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_vram_gb = free_vram / (1024**3)
            # Estimate: ~1GB per frame pair at 1080p
            max_safe_batch = max(1, int(free_vram_gb * 0.6))  # Use 60% of free VRAM
            batch_size = min(batch_size, max_safe_batch, max_batch_for_resolution, total_pairs)
            log.info(f"GIMM-VFI: Using batch_size={batch_size} (resolution: {W}x{H}, max_for_res: {max_batch_for_resolution}, free VRAM: {free_vram_gb:.1f}GB)")

        out_images_list = []
        flows = []
        pbar = ProgressBar(total_pairs)

        autocast_device = mm.get_autocast_device(device)
        cast_context = torch.autocast(device_type=autocast_device, dtype=dtype) if dtype != torch.float32 else nullcontext()

        with torch.no_grad(), cast_context:  # no_grad reduces VRAM by ~20%
            # Process in batches of frame pairs
            for batch_start in range(0, total_pairs, batch_size):
                batch_end = min(batch_start + batch_size, total_pairs)
                current_batch_size = batch_end - batch_start
                
                # Collect frame pairs for this batch
                I0_batch = []
                I2_batch = []
                for j in range(batch_start, batch_end):
                    I0_batch.append(images[j])
                    I2_batch.append(images[j + 1])
                
                # Stack into batches
                I0_stacked = torch.stack(I0_batch, dim=0)  # [B, C, H, W]
                I2_stacked = torch.stack(I2_batch, dim=0)  # [B, C, H, W]
                
                # Add first frame of batch to output (only for first batch)
                if batch_start == 0:
                    out_images_list.append(I0_stacked[0].permute(1, 2, 0).cpu())
                
                # Padding
                padder = InputPadder(I0_stacked.shape, 32)
                I0_padded, I2_padded = padder.pad(I0_stacked, I2_stacked)
                
                # Create batched input: [B, C, 2, H, W]
                xs = torch.cat((I0_padded.unsqueeze(2), I2_padded.unsqueeze(2)), dim=2).to(device, non_blocking=True)
                
                s_shape = xs.shape[-2:]
                
                # Generate coordinate inputs for all interpolation steps
                coord_inputs = [
                    (
                        gimmvfi_model.sample_coord_input(
                            current_batch_size,
                            s_shape,
                            [1 / interpolation_factor * i],
                            device=xs.device,
                            upsample_ratio=ds_factor,
                        ),
                        None,
                    )
                    for i in range(1, interpolation_factor)
                ]
                timesteps = [
                    i * 1 / interpolation_factor * torch.ones(current_batch_size).to(xs.device)
                    for i in range(1, interpolation_factor)
                ]
                
                # Run batched inference
                all_outputs = gimmvfi_model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
                
                # Process outputs for each frame pair in batch
                for b in range(current_batch_size):
                    # Extract interpolated frames for this pair
                    for i, im in enumerate(all_outputs["imgt_pred"]):
                        unpadded = padder.unpad(im[b:b+1])  # Keep batch dim for unpad
                        out_images_list.append(unpadded.squeeze(0).detach().cpu().permute(1, 2, 0))
                        
                        if output_flows and i < len(all_outputs["flowt"]):
                            flowt = padder.unpad(all_outputs["flowt"][i][b:b+1])
                            flowt_img = flow_to_image(
                                flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                                convert_to_bgr=True,
                            )
                            flows.append(flowt_img)
                    
                    # Add the end frame
                    I2_unpadded = padder.unpad(I2_padded[b:b+1])
                    out_images_list.append(I2_unpadded.squeeze(0).detach().cpu().permute(1, 2, 0))
                    
                    pbar.update(1)
                
                # Clean up batch tensors to stabilize VRAM
                del xs, I0_padded, I2_padded, I0_stacked, I2_stacked, all_outputs, coord_inputs, timesteps
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for all GPU ops to complete
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                import gc
                gc.collect()
        
        image_tensors = torch.stack(out_images_list)
        image_tensors = image_tensors.cpu().float()

        rgb_images = [cv2.cvtColor(flow, cv2.COLOR_BGR2RGB) for flow in flows]

        if output_flows:
            flow_tensors = torch.stack([torch.from_numpy(image) for image in rgb_images])
            flow_tensors = flow_tensors / 255.0
            flow_tensors = flow_tensors.cpu().float()
        else:
            flow_tensors = torch.zeros(1, 64, 64, 3)

        return (image_tensors, flow_tensors)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadGIMMVFIModel_A100": DownloadAndLoadGIMMVFIModel,
    "GIMMVFI_interpolate_A100": GIMMVFI_interpolate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadGIMMVFIModel_A100": "(Down)Load GIMMVFI Model [A100]",
    "GIMMVFI_interpolate_A100": "GIMM-VFI Interpolate [A100]",
}
