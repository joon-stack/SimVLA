#!/usr/bin/env python3
"""
SimVLA LIBERO Policy Server (WebSocket)

A WebSocket-based policy server for LIBERO evaluation:
- Uses msgpack_numpy serialization for efficient data transfer
- Sends server metadata on connection
- Receives: observation/image, observation/wrist_image, observation/state, prompt
- Returns: {"actions": [...]}

State format (8D): [ee_pos(3), axis_angle(3), gripper_qpos(2)]
Action format (7D): [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import websockets

try:
    import msgpack
    import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack_numpy not installed, using JSON fallback")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
device = "cuda" if torch.cuda.is_available() else "cpu"
vision_dtype = torch.float32
policy_dtype = torch.float32
amp_dtype: torch.dtype | None = torch.bfloat16 if torch.cuda.is_available() else None
camera_mode = "dual"

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
}


def load_model(
    checkpoint_path: str,
    norm_stats_path: str = None,
    smolvlm_model_path: str = None,
    requested_camera_mode: str = "auto",
):
    """Load SimVLA model and processor."""
    global model, processor, vision_dtype, policy_dtype, camera_mode
    
    logger.info(f"Loading SimVLA from {checkpoint_path}...")
    
    model = SmolVLMVLA.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    vision_dtype = next(model.vlm.model.vision_model.parameters()).dtype
    policy_dtype = next(model.transformer.parameters()).dtype
    config_camera_mode = str(getattr(model.config, "camera_mode", "")).strip().lower()
    if requested_camera_mode == "auto":
        camera_mode = config_camera_mode if config_camera_mode in {"single", "dual"} else "dual"
    else:
        camera_mode = requested_camera_mode
    
    smolvlm_path = smolvlm_model_path or "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_path)
    
    if norm_stats_path and os.path.exists(norm_stats_path):
        logger.info(f"Loading norm stats from: {norm_stats_path}")
        model.action_space.load_norm_stats(norm_stats_path)
        if hasattr(model.action_space, 'state_norm_stats') and model.action_space.state_norm_stats:
            logger.info(f"   State norm: mean={model.action_space.state_norm_stats.mean[:3].tolist()}")
        if hasattr(model.action_space, 'action_norm_stats') and model.action_space.action_norm_stats:
            logger.info(f"   Action norm: mean={model.action_space.action_norm_stats.mean[:3].tolist()}")
    else:
        logger.warning("No norm_stats loaded!")
    
    logger.info(
        "Model loaded! Device: %s, vision_dtype: %s, policy_dtype: %s, amp_dtype: %s, Image size: %sx%s",
        device,
        vision_dtype,
        policy_dtype,
        amp_dtype,
        CONFIG["image_size"],
        CONFIG["image_size"],
    )
    logger.info("Evaluation camera_mode: %s (config=%s)", camera_mode, config_camera_mode or "missing")


def _autocast_context():
    if device == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def preprocess_images(image0: np.ndarray, image1: np.ndarray | None, active_camera_mode: str):
    """Preprocess images to model input format."""
    image_size = CONFIG["image_size"]
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img0 = Image.fromarray(image0.astype(np.uint8))
    img0_t = transform(img0)
    padding = torch.zeros_like(img0_t)

    if active_camera_mode == "single":
        images = torch.stack([img0_t, padding, padding.clone()], dim=0)
        image_mask = torch.tensor([[True, False, False]])
        return images.unsqueeze(0), image_mask

    if image1 is None:
        images = torch.stack([img0_t, padding, padding.clone()], dim=0)
        image_mask = torch.tensor([[True, False, False]])
        return images.unsqueeze(0), image_mask

    img1 = Image.fromarray(image1.astype(np.uint8))
    img1_t = transform(img1)

    # Pad to 3 views (model processes all views together)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])

    return images.unsqueeze(0), image_mask


def decode_numpy(obj):
    """Decode numpy array from msgpack_numpy dict format."""
    if isinstance(obj, dict):
        if b'__ndarray__' in obj or '__ndarray__' in obj:
            data_key = b'data' if b'data' in obj else 'data'
            dtype_key = b'dtype' if b'dtype' in obj else 'dtype'
            shape_key = b'shape' if b'shape' in obj else 'shape'
            
            data = obj[data_key]
            dtype_str = obj[dtype_key]
            shape = obj[shape_key]
            
            if isinstance(dtype_str, bytes):
                dtype_str = dtype_str.decode()
            
            if shape and isinstance(shape[0], bytes):
                shape = tuple(int(s) for s in shape)
            else:
                shape = tuple(shape)
            
            return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
    return obj


def infer(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single observation."""
    global model, processor, vision_dtype, policy_dtype, amp_dtype, camera_mode
    
    try:
        # Extract observation fields
        image0 = observation.get("observation/image")
        image1 = observation.get("observation/wrist_image")
        state = observation.get("observation/state", np.zeros(8))
        prompt = observation.get("prompt", "")
        
        # Decode msgpack_numpy format if needed
        image0 = decode_numpy(image0)
        image1 = decode_numpy(image1)
        state = decode_numpy(state)
        
        # Ensure numpy arrays
        if not isinstance(image0, np.ndarray):
            image0 = np.array(image0, dtype=np.uint8)
        if image1 is not None and not isinstance(image1, np.ndarray):
            image1 = np.array(image1, dtype=np.uint8)
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        state = state[:8]
        
        # Preprocess images
        images, image_mask = preprocess_images(image0, image1, camera_mode)
        images = images.to(device)
        image_mask = image_mask.to(device)
        
        # Encode language instruction
        lang = processor.encode_language([prompt])
        lang = {k: v.to(device) for k, v in lang.items()}
        
        # Proprioception
        proprio_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad(), _autocast_context():
            actions = model.generate_actions(
                input_ids=lang['input_ids'],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio_tensor,
                steps=CONFIG["action_horizon"],
            )
        
        actions = actions.cpu().numpy()[0]
        
        return {"actions": actions}
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        traceback.print_exc()
        return {"actions": np.zeros((CONFIG["action_horizon"], CONFIG["action_dim"]))}


async def handle_connection(websocket, path=None):
    """Handle a WebSocket connection."""
    logger.info(f"Connection from {websocket.remote_address} opened")
    
    try:
        # Send server metadata on connection
        metadata = {
            "model": "SimVLA",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "image_size": CONFIG["image_size"],
        }
        if HAS_MSGPACK:
            await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
        else:
            import json
            await websocket.send(json.dumps(metadata))
        
        # Process requests
        async for message in websocket:
            try:
                # Parse request
                if HAS_MSGPACK and isinstance(message, bytes):
                    request = msgpack_numpy.unpackb(message, raw=False)
                else:
                    import json
                    request = json.loads(message)
                
                # Run inference
                result = infer(request)
                
                # Send response (convert numpy to list for compatibility)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                
                response_data = {"actions": actions}
                
                if HAS_MSGPACK:
                    import msgpack
                    response = msgpack.packb(response_data, use_bin_type=True)
                else:
                    import json
                    response = json.dumps(response_data)
                
                await websocket.send(response)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                error_msg = f"Error: {str(e)}"
                await websocket.send(error_msg)
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(f"Connection from {websocket.remote_address} closed")


async def serve(host: str, port: int):
    """Start the WebSocket server."""
    logger.info(f"Creating SimVLA server (host: {host}, port: {port})")
    
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SimVLA LIBERO Server (WebSocket)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SimVLA checkpoint")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to normalization stats JSON")
    parser.add_argument("--smolvlm_model", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM model path or HuggingFace repo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help="Autocast precision for inference. 'bf16' matches training scripts.")
    parser.add_argument("--camera_mode", type=str, default="auto", choices=["auto", "single", "dual"],
                        help="Camera views to use for eval. 'auto' reads checkpoint config when available.")
    
    args = parser.parse_args()
    global amp_dtype
    if args.mixed_precision == "no":
        amp_dtype = None
    elif args.mixed_precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.bfloat16
    
    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")
    
    load_model(args.checkpoint, args.norm_stats, args.smolvlm_model, args.camera_mode)
    
    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
