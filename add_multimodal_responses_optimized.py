#!/usr/bin/env python3
"""
Optimized MM-SafetyBench Multimodal Evaluation Script

This optimized script supports multiple vision-language models while maintaining:
- Performance and memory efficiency
- Code organization and maintainability  
- Error handling and resource management
- Constants and configuration management

Supported Models:
- CogVLM (THUDM)
- CogVLM2 (THUDM) 
- InstructBLIP (Salesforce)
- Qwen-VL (Alibaba)
"""

import json
import os
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache

warnings.filterwarnings("ignore")

# Constants
DEFAULT_DATA_DIR = "../data/MM-SafetyBench"
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7

# MM-SafetyBench Categories (centralized configuration)
SAFETY_CATEGORIES = [
    "01-Illegal_Activitiy", "02-HateSpeech", "03-Malware_Generation",
    "04-Physical_Harm", "05-EconomicHarm", "06-Fraud", "07-Sex",
    "08-Political_Lobbying", "09-Privacy_Violence", "10-Legal_Opinion",
    "11-Financial_Advice", "12-Health_Consultation", "13-Gov_Decision"
]

# Image-Question Kind Mappings
KIND_MAPPINGS = {
    "kind1": ("SD", "Rephrased Question(SD)"),
    "kind2": ("SD_TYPO", "Rephrased Question"),
    "kind3": ("TYPO", "Rephrased Question")
}

@dataclass
class ModelConfig:
    """Configuration for multimodal models."""
    display_name: str
    description: str
    hf_name: str
    architecture: str
    model_type: str
    requires_trust_remote_code: bool = False

# Available multimodal models (using dataclass for better organization)
AVAILABLE_MODELS = {
    "cogvlm-chat-17b": ModelConfig(
        display_name="CogVLM-Chat(17B)",
        description="CogVLM conversational model with 17B parameters (high performance, more memory)",
        hf_name="THUDM/cogvlm-chat-hf",
        architecture="cogvlm",
        model_type="chat",
        requires_trust_remote_code=True
    ),
    "cogvlm2-llama3-chat-19b": ModelConfig(
        display_name="CogVLM2-Llama3-Chat(19B)",
        description="CogVLM2 with Llama3 backbone, 19B parameters (latest, best performance)",
        hf_name="THUDM/cogvlm2-llama3-chat-19B",
        architecture="cogvlm2",
        model_type="chat",
        requires_trust_remote_code=True
    ),
    "instructblip-vicuna-7b": ModelConfig(
        display_name="InstructBLIP-Vicuna(7B)",
        description="InstructBLIP with Vicuna 7B backbone (instruction-following, moderate size)",
        hf_name="Salesforce/instructblip-vicuna-7b",
        architecture="instructblip",
        model_type="instruct",
        requires_trust_remote_code=False
    ),
    "instructblip-vicuna-13b": ModelConfig(
        display_name="InstructBLIP-Vicuna(13B)",
        description="InstructBLIP with Vicuna 13B backbone (instruction-following, larger)",
        hf_name="Salesforce/instructblip-vicuna-13b",
        architecture="instructblip",
        model_type="instruct",
        requires_trust_remote_code=False
    ),
    "qwen-vl-chat": ModelConfig(
        display_name="Qwen-VL-Chat",
        description="Qwen-VL conversational model (multilingual, efficient)",
        hf_name="Qwen/Qwen-VL-Chat",
        architecture="qwen-vl",
        model_type="chat",
        requires_trust_remote_code=True
    ),
    "qwen-vl": ModelConfig(
        display_name="Qwen-VL",
        description="Qwen-VL base model (multilingual, instruction-following)",
        hf_name="Qwen/Qwen-VL",
        architecture="qwen-vl",
        model_type="base",
        requires_trust_remote_code=True
    )
}


class OptimizedPathManager:
    """Centralized path management for better organization."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.questions_dir = self.data_dir / "processed_questions"
        self.images_dir = self.data_dir / "imgs"
        self.answer_format_dir = self.data_dir / "answer_format"
        self.results_dir = self.data_dir / "question_and_answer"
    
    def get_question_file(self, category: str) -> Path:
        """Get path to question file for a category."""
        return self.questions_dir / f"{category}.json"
    
    def get_image_path(self, category: str, kind: str, question_id: str) -> Path:
        """Get path to image file."""
        img_dir = KIND_MAPPINGS[kind][0]
        return self.images_dir / category / img_dir / f"{question_id}.jpg"
    
    def get_output_dir(self, model_name: str, kind: str) -> Path:
        """Get output directory for model results."""
        img_dir = KIND_MAPPINGS[kind][0]
        return self.results_dir / model_name / img_dir
    
    def get_output_file(self, model_name: str, kind: str, category: str) -> Path:
        """Get output file path for results."""
        return self.get_output_dir(model_name, kind) / f"{category}_with_responses.json"


class ModelResponseProcessor:
    """Handles response processing and extraction."""
    
    @staticmethod
    def extract_response_text(response) -> str:
        """Extract text from model response."""
        if isinstance(response, tuple):
            if len(response) > 0:
                return str(response[0])
            else:
                return "Error: Empty response tuple"
        return str(response)


class OptimizedMultimodalQueryEngine:
    """Optimized multimodal query engine supporting multiple model architectures."""
    
    def __init__(self, model_key: str = "instructblip-vicuna-7b"):
        """Initialize multimodal model for querying."""
        self.model_config = self._get_model_config(model_key)
        self.model_key = model_key
        self.display_name = self.model_config.display_name
        self.hf_model_name = self.model_config.hf_name
        self.architecture = self.model_config.architecture
        
        # Model components (lazy loading)
        self.processor = None
        self.model = None
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Selected model: {self.display_name}")
        print(f"Architecture: {self.architecture}")
    
    @staticmethod
    def _get_model_config(model_key: str) -> ModelConfig:
        """Get model configuration with validation."""
        if model_key not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")
        return AVAILABLE_MODELS[model_key]
    
    def _load_cogvlm(self) -> None:
        """Load CogVLM model."""
        try:
            from transformers import LlamaTokenizer, AutoModelForCausalLM
            
            print(f"Loading CogVLM tokenizer and model...")
            # Step 1: Load the tokenizer from vicuna (CogVLM doesn't have its own tokenizer)
            print("Loading LlamaTokenizer from vicuna-7b-v1.5...")
            self.processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
            
            # Step 2: Load CogVLM model (this will use the custom classes via auto_map)
            print("Loading CogVLM model with trust_remote_code=True...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=self.model_config.requires_trust_remote_code,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
            
            print("CogVLM model loaded successfully!")
            
        except ImportError as e:
            raise ImportError(f"CogVLM requires transformers with LlamaTokenizer: {e}")
        except Exception as e:
            print(f"Error loading CogVLM: {e}")
            # If the main approach fails, try a more explicit method
            try:
                print("Trying explicit CogVLM loading...")
                import sys
                import os
                
                # Add the model's custom modules to Python path temporarily
                model_cache_path = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/THUDM/cogvlm-chat-hf")
                if os.path.exists(model_cache_path) and model_cache_path not in sys.path:
                    sys.path.insert(0, model_cache_path)
                
                from transformers import LlamaTokenizer, AutoModelForCausalLM
                
                # Load tokenizer
                self.processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
                
                # Load model with explicit trust_remote_code
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hf_model_name,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,  # Explicitly set to True
                    device_map="auto" if self.device == "cuda" else None,
                    local_files_only=False  # Allow downloading if needed
                ).eval()
                
                print("CogVLM loaded successfully with explicit method!")
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load CogVLM. Primary error: {e}. Fallback error: {e2}")
    
    def _load_cogvlm2(self) -> None:
        """Load CogVLM2 model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading CogVLM2 tokenizer and model...")
            self.processor = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                trust_remote_code=self.model_config.requires_trust_remote_code
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=self.model_config.requires_trust_remote_code,
                device_map="auto" if self.device == "cuda" else None
            ).eval()
            
        except ImportError as e:
            raise ImportError(f"CogVLM2 requires specific dependencies: {e}")
    
    def _load_instructblip(self) -> None:
        """Load InstructBLIP model."""
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            
            print(f"Loading InstructBLIP processor and model...")
            self.processor = InstructBlipProcessor.from_pretrained(self.hf_model_name)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
        except ImportError as e:
            raise ImportError(f"InstructBLIP requires transformers: {e}")
    
    def _load_qwen_vl(self) -> None:
        """Load Qwen-VL model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading Qwen-VL tokenizer and model...")
            self.processor = AutoTokenizer.from_pretrained(
                self.hf_model_name,
                trust_remote_code=self.model_config.requires_trust_remote_code
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=self.model_config.requires_trust_remote_code,
                fp16=True if self.device == "cuda" else False
            ).eval()
            
        except ImportError as e:
            raise ImportError(f"Qwen-VL requires specific dependencies: {e}")
    
    def load_model(self) -> None:
        """Load the appropriate model based on architecture."""
        try:
            print(f"Loading {self.display_name}")
            print(f"HuggingFace model: {self.hf_model_name}")
            print(f"Using device: {self.device}")
            
            if self.architecture == "cogvlm":
                self._load_cogvlm()
            elif self.architecture == "cogvlm2":
                self._load_cogvlm2()
            elif self.architecture == "instructblip":
                self._load_instructblip()
            elif self.architecture == "qwen-vl":
                self._load_qwen_vl()
            else:
                raise ValueError(f"Unsupported architecture: {self.architecture}")
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image with error handling."""
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"Loaded image: {image_path}")
            return image
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            return None
    
    def _preprocess_image_for_cogvlm(self, image: Image.Image) -> Image.Image:
        """Preprocess image for CogVLM with proper sizing."""
        # According to config.json, CogVLM expects 490x490 images
        target_size = 490
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image to the expected size
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image

    def _apply_cogvlm_compatibility_patch(self):
        """Apply comprehensive compatibility patches for CogVLM with newer transformers."""
        if not hasattr(self.model, '_cogvlm_patched'):
            print("Applying CogVLM compatibility patches...")
            
            # CRITICAL FIX: Disable past_key_values completely to fix API compatibility
            if hasattr(self.model, '_extract_past_from_model_output') and not hasattr(self.model, '_extract_past_patched'):
                # Store original method and mark as patched to prevent recursion
                self.model._original_extract_past = self.model._extract_past_from_model_output
                self.model._extract_past_patched = True
                
                def no_past_extract(outputs, **kwargs):
                    # Always return None to disable past_key_values completely
                    # This fixes the 'standardize_cache_format' API incompatibility
                    return None
                
                self.model._extract_past_from_model_output = no_past_extract
                print("🔧 Disabled past_key_values extraction (API compatibility fix)")
            
            # CRITICAL FIX: Clean prepare_inputs_for_generation outputs
            if hasattr(self.model, 'prepare_inputs_for_generation') and not hasattr(self.model, '_prepare_inputs_patched'):
                # Store original method and mark as patched to prevent recursion
                self.model._original_prepare_inputs = self.model.prepare_inputs_for_generation
                self.model._prepare_inputs_patched = True
                
                def cleaned_prepare_inputs(*args, **kwargs):
                    # Call original method first
                    result = self.model._original_prepare_inputs(*args, **kwargs)
                    
                    # Clean the result to remove any problematic values
                    cleaned_result = {}
                    for key, value in result.items():
                        if isinstance(value, str):
                            print(f"🔧 Filtering string from prepare_inputs: {key}")
                            continue
                        elif isinstance(value, (list, tuple)):
                            cleaned_list = [item for item in value if not isinstance(item, str)]
                            if cleaned_list:
                                cleaned_result[key] = cleaned_list
                        else:
                            cleaned_result[key] = value
                    
                    return cleaned_result
                
                self.model.prepare_inputs_for_generation = cleaned_prepare_inputs
                print("🔧 Enhanced prepare_inputs_for_generation")
            
            # CRITICAL FIX: Override generate method with comprehensive fixes
            if hasattr(self.model, 'generate') and not hasattr(self.model, '_generate_patched'):
                # Store original method and mark as patched to prevent recursion
                self.model._original_generate = self.model.generate
                self.model._generate_patched = True
                
                def fixed_generate(*args, **kwargs):
                    # Force disable caching to prevent API issues
                    kwargs['use_cache'] = False
                    kwargs['past_key_values'] = None
                    
                    # Call the original unpatched method directly
                    return self.model._original_generate(*args, **kwargs)
                
                self.model.generate = fixed_generate
                print("🔧 Enhanced generate method with cache disabling")
            
            # NEW: Patch the vision model's patch embedding to handle tensor size mismatches
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'vision'):
                vision_model = self.model.model.vision
                if hasattr(vision_model, 'patch_embedding'):
                    patch_embedding = vision_model.patch_embedding
                    
                    # Store original forward method
                    if hasattr(patch_embedding, 'forward') and not hasattr(patch_embedding, '_original_forward'):
                        patch_embedding._original_forward = patch_embedding.forward
                        
                        def patched_patch_embedding_forward(x):
                            try:
                                return patch_embedding._original_forward(x)
                            except RuntimeError as e:
                                if "Sizes of tensors must match except in dimension 1" in str(e):
                                    print(f"🔧 Fixing patch embedding tensor size mismatch: {e}")
                                    
                                    # Handle the tensor size mismatch issue
                                    # Get the conv projection
                                    if hasattr(patch_embedding, 'proj'):
                                        proj = patch_embedding.proj
                                        
                                        # Apply convolution
                                        x_conv = proj(x)  # [batch, channels, h, w]
                                        B, C, H, W = x_conv.shape
                                        
                                        # Reshape to sequence
                                        x_seq = x_conv.reshape(B, C, H * W).transpose(1, 2)  # [batch, seq_len, channels]
                                        
                                        # Handle CLS token
                                        if hasattr(patch_embedding, 'cls_embedding'):
                                            cls_token = patch_embedding.cls_embedding
                                            cls_expanded = cls_token.expand(B, -1, -1)
                                            
                                            # Handle position embeddings
                                            if hasattr(patch_embedding, 'position_embedding'):
                                                pos_emb = patch_embedding.position_embedding
                                                
                                                # Calculate expected sequence length with CLS token
                                                expected_len = x_seq.shape[1] + 1  # +1 for CLS token
                                                available_len = pos_emb.weight.shape[0]
                                                
                                                print(f"🔧 Patch embedding debug:")
                                                print(f"   Input image: {x.shape}")
                                                print(f"   After conv: {x_conv.shape}")
                                                print(f"   Sequence: {x_seq.shape}")
                                                print(f"   CLS token: {cls_expanded.shape}")
                                                print(f"   Expected pos emb length: {expected_len}")
                                                print(f"   Available pos emb length: {available_len}")
                                                
                                                if expected_len <= available_len:
                                                    # Normal case - we have enough position embeddings
                                                    x_with_cls = torch.cat((cls_expanded, x_seq), dim=1)
                                                    pos_embeddings = pos_emb.weight[:expected_len].unsqueeze(0).expand(B, -1, -1)
                                                    x_final = x_with_cls + pos_embeddings
                                                else:
                                                    # Fallback - truncate or interpolate position embeddings
                                                    print(f"🔧 Position embedding mismatch - using fallback strategy")
                                                    
                                                                                                         # Try to interpolate position embeddings
                                                    
                                                    # Separate CLS and patch position embeddings
                                                    cls_pos = pos_emb.weight[0:1]  # CLS position
                                                    patch_pos = pos_emb.weight[1:]  # Patch positions
                                                    
                                                    # Calculate grid size for interpolation
                                                    old_grid_size = int((patch_pos.shape[0]) ** 0.5)
                                                    new_grid_size = int((x_seq.shape[1]) ** 0.5)
                                                    
                                                    if old_grid_size * old_grid_size == patch_pos.shape[0]:
                                                        # Reshape and interpolate
                                                        patch_pos_2d = patch_pos.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
                                                        patch_pos_new = F.interpolate(
                                                            patch_pos_2d, 
                                                            size=(new_grid_size, new_grid_size), 
                                                            mode='bilinear', 
                                                            align_corners=False
                                                        )
                                                        patch_pos_new = patch_pos_new.permute(0, 2, 3, 1).reshape(-1, patch_pos.shape[-1])
                                                        
                                                        # Combine CLS and interpolated patch positions
                                                        new_pos_emb = torch.cat([cls_pos, patch_pos_new], dim=0)
                                                        
                                                        # Apply to sequence
                                                        x_with_cls = torch.cat((cls_expanded, x_seq), dim=1)
                                                        pos_embeddings = new_pos_emb.unsqueeze(0).expand(B, -1, -1)
                                                        x_final = x_with_cls + pos_embeddings
                                                        
                                                        print(f"🔧 Successfully interpolated position embeddings: {old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size}")
                                                    else:
                                                        # Fallback to truncation/padding
                                                        print(f"🔧 Using truncation fallback")
                                                        x_with_cls = torch.cat((cls_expanded, x_seq), dim=1)
                                                        
                                                        if expected_len <= available_len:
                                                            pos_embeddings = pos_emb.weight[:expected_len].unsqueeze(0).expand(B, -1, -1)
                                                        else:
                                                            # Pad with zeros
                                                            padding_needed = expected_len - available_len
                                                            pos_base = pos_emb.weight
                                                            pos_padding = torch.zeros(padding_needed, pos_base.shape[1], 
                                                                                   dtype=pos_base.dtype, device=pos_base.device)
                                                            pos_embeddings = torch.cat([pos_base, pos_padding], dim=0).unsqueeze(0).expand(B, -1, -1)
                                                        
                                                        x_final = x_with_cls + pos_embeddings
                                            else:
                                                # No position embeddings - just concatenate
                                                x_final = torch.cat((cls_expanded, x_seq), dim=1)
                                        else:
                                            # No CLS token - just return sequence
                                            x_final = x_seq
                                            
                                            if hasattr(patch_embedding, 'position_embedding'):
                                                pos_emb = patch_embedding.position_embedding
                                                seq_len = x_seq.shape[1]
                                                if seq_len <= pos_emb.weight.shape[0]:
                                                    pos_embeddings = pos_emb.weight[:seq_len].unsqueeze(0).expand(B, -1, -1)
                                                    x_final = x_seq + pos_embeddings
                                        
                                        print(f"🔧 Patch embedding fix successful: {x_final.shape}")
                                        return x_final
                                    else:
                                        print(f"🔧 No conv projection found in patch embedding")
                                        raise e
                                else:
                                    raise e
                        
                        patch_embedding.forward = patched_patch_embedding_forward
                        print("🔧 Applied patch embedding dimension fix")
            
            # Patch the _extract_past_from_model_output method
            if hasattr(self.model, '_extract_past_from_model_output'):
                original_extract_past = self.model._extract_past_from_model_output
                
                def patched_extract_past(outputs, standardize_cache_format=None, **kwargs):
                    try:
                        # Remove the incompatible parameter
                        return original_extract_past(outputs, **kwargs)
                    except TypeError as e:
                        if "standardize_cache_format" in str(e):
                            # Fallback: return None to disable past key values
                            return None
                        raise e
                
                self.model._extract_past_from_model_output = patched_extract_past
            
            # Patch the llm_forward method to handle string past_key_values
            if hasattr(self.model, 'llm_forward'):
                original_llm_forward = self.model.llm_forward
                
                def patched_llm_forward(input_ids, past_key_values=None, **kwargs):
                    try:
                        # Check if past_key_values contains strings instead of tensors
                        if past_key_values is not None:
                            if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
                                if isinstance(past_key_values[0], str) or \
                                   (isinstance(past_key_values[0], (list, tuple)) and len(past_key_values[0]) > 0 and isinstance(past_key_values[0][0], str)):
                                    # Set past_key_values to None if it contains strings
                                    past_key_values = None
                                    print("Warning: Converted string past_key_values to None for compatibility")
                        
                        return original_llm_forward(input_ids, past_key_values=past_key_values, **kwargs)
                    except AttributeError as e:
                        if "'str' object has no attribute 'shape'" in str(e):
                            # Retry without past_key_values
                            print("Warning: Retrying without past_key_values due to string/tensor incompatibility")
                            return original_llm_forward(input_ids, past_key_values=None, **kwargs)
                        raise e
                
                self.model.llm_forward = patched_llm_forward
            
            # NEW: Patch prepare_inputs_for_generation to handle token_type_ids properly
            if hasattr(self.model, 'prepare_inputs_for_generation'):
                original_prepare_inputs = self.model.prepare_inputs_for_generation
                
                def patched_prepare_inputs(input_ids, past_key_values=None, attention_mask=None, token_type_ids=None, **kwargs):
                    try:
                        # Ensure token_type_ids exists if the model requires it
                        if token_type_ids is None:
                            # Create default token_type_ids based on input_ids
                            batch_size, seq_len = input_ids.shape
                            token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
                            print("Generated default token_type_ids for CogVLM compatibility")
                        
                        return original_prepare_inputs(
                            input_ids=input_ids, 
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            **kwargs
                        )
                    except TypeError as e:
                        if "token_type_ids" in str(e):
                            print(f"Warning: token_type_ids issue in prepare_inputs_for_generation: {e}")
                            # Try without token_type_ids
                            try:
                                return original_prepare_inputs(
                                    input_ids=input_ids,
                                    past_key_values=past_key_values,
                                    attention_mask=attention_mask,
                                    **kwargs
                                )
                            except Exception as e2:
                                print(f"Fallback prepare_inputs also failed: {e2}")
                                # Return minimal dict
                                return {
                                    'input_ids': input_ids,
                                    'attention_mask': attention_mask,
                                    'past_key_values': past_key_values,
                                    'token_type_ids': torch.zeros_like(input_ids, dtype=torch.long) if token_type_ids is None else token_type_ids
                                }
                        raise e
                
                self.model.prepare_inputs_for_generation = patched_prepare_inputs
            
            # NEW: Alternative approach - patch the generate method more directly
            if hasattr(self.model, 'generate'):
                original_generate = self.model.generate
                
                def patched_generate(*args, **kwargs):
                    try:
                        # Ensure we have token_type_ids if using positional input_ids
                        if len(args) > 0 and 'token_type_ids' not in kwargs:
                            input_ids = args[0]
                            if torch.is_tensor(input_ids):
                                kwargs['token_type_ids'] = torch.zeros_like(input_ids, dtype=torch.long)
                        
                        return original_generate(*args, **kwargs)
                    except TypeError as e:
                        if "token_type_ids" in str(e) and "missing" in str(e):
                            print(f"Token type IDs missing error: {e}")
                            # Add token_type_ids and retry
                            if len(args) > 0:
                                input_ids = args[0]
                                if torch.is_tensor(input_ids):
                                    kwargs['token_type_ids'] = torch.zeros_like(input_ids, dtype=torch.long)
                            return original_generate(*args, **kwargs)
                        raise e
                
                self.model.generate = patched_generate
            
            # Mark as patched
            self.model._cogvlm_patched = True
            print("CogVLM compatibility patches applied successfully!")

    def _cogvlm_simple_fallback(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Simple text-only fallback for CogVLM when multimodal processing fails."""
        try:
            print("Using CogVLM simple text-only fallback...")
            # Simple text-only tokenization
            query = f'Question: {question} Answer:'
            inputs = self.processor(query, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.eos_token_id,
                    eos_token_id=self.processor.eos_token_id,
                    use_cache=False
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            if 'Answer:' in response:
                response = response.split('Answer:')[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Fallback Error: {str(e)}"

    def _clean_cogvlm_inputs(self, inputs: Dict) -> Dict:
        """Clean and validate CogVLM inputs to ensure only proper tensors are passed."""
        cleaned_inputs = {}
        
        for key, value in inputs.items():
            # Skip known problematic string fields
            if isinstance(value, str):
                print(f"Debug: Skipping string field '{key}': {value[:50]}..." if len(str(value)) > 50 else f"Debug: Skipping string field '{key}': {value}")
                continue
                
            # Handle list/tuple fields (like images)
            elif isinstance(value, (list, tuple)):
                cleaned_list = []
                for item in value:
                    if torch.is_tensor(item):
                        # Ensure proper dimensions and device
                        if item.dim() == 3:
                            item = item.unsqueeze(0)  # Add batch dimension
                        item = item.to(self.device)
                        # Convert to bfloat16 if model uses bfloat16 (regardless of device)
                        if item.dtype == torch.float32:
                            item = item.to(torch.bfloat16)
                        cleaned_list.append(item)
                    elif isinstance(item, str):
                        print(f"Debug: Skipping string item in '{key}': {item[:50]}..." if len(str(item)) > 50 else f"Debug: Skipping string item in '{key}': {item}")
                        continue
                    else:
                        cleaned_list.append(item)
                
                if cleaned_list:  # Only add if we have valid items
                    cleaned_inputs[key] = cleaned_list
                    
            # Handle tensor fields
            elif torch.is_tensor(value):
                if value.dim() == 1:
                    value = value.unsqueeze(0)  # Add batch dimension
                value = value.to(self.device)
                cleaned_inputs[key] = value
                
            # Skip other non-tensor types that might cause issues
            elif hasattr(value, 'shape'):
                cleaned_inputs[key] = value
            else:
                print(f"Debug: Skipping non-tensor field '{key}' of type {type(value)}")
                
        return cleaned_inputs

    def _query_cogvlm(self, question: str, image: Optional[Image.Image] = None, 
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query CogVLM model with improved error handling and input validation."""
        try:
            # Apply compatibility patches
            self._apply_cogvlm_compatibility_patch()
            
            # Preprocess image if provided
            if image is not None:
                image = self._preprocess_image_for_cogvlm(image)
                # CogVLM uses <EOI> token to separate image and text
                query = f'<EOI>Question: {question} Answer:'
            else:
                # Text-only query
                query = f'Question: {question} Answer:'
            
            print(f"Processing query: {query[:100]}...")
            
            # Prepare inputs with enhanced validation and fallback
            try:
                if image is not None:
                    print("Building conversation inputs with image...")
                    inputs = self.model.build_conversation_input_ids(
                        self.processor, 
                        query=query, 
                        history=[], 
                        images=[image]
                    )
                else:
                    print("Building conversation inputs (text-only)...")
                    inputs = self.model.build_conversation_input_ids(
                        self.processor, 
                        query=query, 
                        history=[], 
                        images=[]
                    )
                print(f"CogVLM returned input keys: {list(inputs.keys())}")
                
            except Exception as e:
                print(f"Error in build_conversation_input_ids: {e}")
                print("Using simplified tokenization fallback...")
                # Fallback: simple tokenization
                inputs = self.processor(query, return_tensors='pt')
                if image is not None:
                    print("Warning: Image processing failed, using text-only fallback")
            
            # Clean inputs using the new robust cleaning method
            print("Cleaning CogVLM inputs...")
            processed_inputs = self._clean_cogvlm_inputs(inputs)
            
            # Ensure we have the minimal required inputs
            if 'input_ids' not in processed_inputs:
                return "Error: No valid input_ids found after processing"
            
            # Add token_type_ids if missing (CogVLM often requires this)
            if 'token_type_ids' not in processed_inputs and 'input_ids' in processed_inputs:
                seq_len = processed_inputs['input_ids'].shape[1]
                batch_size = processed_inputs['input_ids'].shape[0]
                processed_inputs['token_type_ids'] = torch.zeros(batch_size, seq_len, dtype=torch.long).to(self.device)
                print(f"Generated token_type_ids with shape: {processed_inputs['token_type_ids'].shape}")
            
            # Generate response with multiple fallback strategies
            print("Generating response...")
            with torch.no_grad():
                try:
                    # First attempt: Full generation with all inputs
                    outputs = self.model.generate(
                        **processed_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=DEFAULT_TEMPERATURE,
                        pad_token_id=self.processor.eos_token_id,
                        eos_token_id=self.processor.eos_token_id,
                        use_cache=False,  # Disable caching to avoid compatibility issues
                        return_dict_in_generate=False  # Simpler output format
                    )
                    
                except Exception as e:
                    print(f"Primary generation attempt failed: {e}")
                    
                    # The comprehensive patches should handle most issues, but provide fallback
                    if "standardize_cache_format" in str(e):
                        print("🔧 API compatibility issue detected - patches should have fixed this")
                        
                    print("Trying simplified generation...")
                    try:
                        # Simplified generation with minimal parameters
                        outputs = self.model.generate(
                            processed_inputs['input_ids'],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=False,
                            pad_token_id=self.processor.eos_token_id
                        )
                        
                    except Exception as e2:
                        print(f"Simplified generation also failed: {e2}")
                        print("All generation attempts failed - using text-only fallback...")
                        return self._cogvlm_simple_fallback(question, max_new_tokens)
            
            # Decode response
            if outputs is None or (torch.is_tensor(outputs) and outputs.numel() == 0):
                return "Error: Model returned empty output"
            
            if torch.is_tensor(outputs):
                # Extract only new tokens (excluding input)
                input_length = processed_inputs.get('input_ids', torch.tensor([[]])).shape[1]
                if input_length > 0 and outputs.shape[1] > input_length:
                    new_tokens = outputs[0][input_length:]
                    response = self.processor.decode(new_tokens, skip_special_tokens=True)
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                response = str(outputs)
            
            # Clean up response format
            if 'Answer:' in response:
                response = response.split('Answer:')[-1].strip()
            
            # Remove any remaining special tokens or artifacts
            response = response.replace('<EOI>', '').strip()
            
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in CogVLM query: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            # Intelligent fallback based on error type
            if "tensor" in str(e).lower() and image is not None:
                print("Tensor error with image - trying text-only fallback...")
                return self._query_cogvlm(question, None, max_new_tokens)
            elif "configuration" in str(e).lower() or "tokenizer" in str(e).lower():
                print("Configuration error - trying simple fallback...")
                return self._cogvlm_simple_fallback(question, max_new_tokens)
            elif "shape" in str(e).lower() or "str" in str(e).lower():
                print("Shape/string error - trying simple fallback...")
                return self._cogvlm_simple_fallback(question, max_new_tokens)
            else:
                print("Unknown error - trying simple fallback...")
                return self._cogvlm_simple_fallback(question, max_new_tokens)
    
    def _query_cogvlm2(self, question: str, image: Optional[Image.Image] = None,
                      max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query CogVLM2 model."""
        try:
            if image is not None:
                # CogVLM2 with image
                query = f'<EOI>Question: {question} Answer:'
                inputs = self.model.build_conversation_input_ids(
                    self.processor,
                    query=query,
                    history=[],
                    images=[image]
                )
            else:
                # CogVLM2 text-only
                query = f'Question: {question} Answer:'
                inputs = self.processor(query, return_tensors='pt')
            
            # Only move tensors to device, leave other objects as-is
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=DEFAULT_TEMPERATURE,
                    pad_token_id=self.processor.eos_token_id
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            if 'Answer:' in response:
                response = response.split('Answer:')[-1].strip()
            
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in CogVLM2 query: {e}")
            return f"Error: {str(e)}"
    
    def _query_instructblip(self, question: str, image: Optional[Image.Image] = None,
                           max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query InstructBLIP model."""
        try:
            if image is not None:
                # InstructBLIP with image
                inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
            else:
                # InstructBLIP text-only (fallback)
                inputs = self.processor(text=question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=DEFAULT_TEMPERATURE,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in InstructBLIP query: {e}")
            return f"Error: {str(e)}"
    
    def _query_qwen_vl(self, question: str, image: Optional[Image.Image] = None,
                      max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query Qwen-VL model."""
        try:
            if image is not None:
                # Qwen-VL with image - save temp image for Qwen format
                temp_image_path = "/tmp/temp_image.jpg"
                image.save(temp_image_path)
                
                if self.model_config.model_type == "chat":
                    query = f'<img>{temp_image_path}</img>{question}'
                else:
                    query = f'Picture 1: <img>{temp_image_path}</img>\n{question}'
            else:
                # Qwen-VL text-only
                query = question
            
            inputs = self.processor(query, return_tensors='pt')
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=DEFAULT_TEMPERATURE,
                    pad_token_id=self.processor.pad_token_id
                )
            
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up temp file
            if image is not None and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in Qwen-VL query: {e}")
            return f"Error: {str(e)}"
    
    def query_text_only(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query model with text only (no image)."""
        if self.model is None:
            self.load_model()
        
        try:
            if self.architecture == "cogvlm":
                return self._query_cogvlm(question, None, max_new_tokens)
            elif self.architecture == "cogvlm2":
                return self._query_cogvlm2(question, None, max_new_tokens)
            elif self.architecture == "instructblip":
                return self._query_instructblip(question, None, max_new_tokens)
            elif self.architecture == "qwen-vl":
                return self._query_qwen_vl(question, None, max_new_tokens)
            else:
                return f"Error: Unsupported architecture {self.architecture}"
                
        except Exception as e:
            print(f"Error in query_text_only: {e}")
            return f"Error: {str(e)}"
    
    def query_with_image(self, question: str, image_path: Path, 
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query model with image and text."""
        if self.model is None:
            self.load_model()
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            print("Image not found, falling back to text-only...")
            return self.query_text_only(question, max_new_tokens)
        
        try:
            if self.architecture == "cogvlm":
                return self._query_cogvlm(question, image, max_new_tokens)
            elif self.architecture == "cogvlm2":
                return self._query_cogvlm2(question, image, max_new_tokens)
            elif self.architecture == "instructblip":
                return self._query_instructblip(question, image, max_new_tokens)
            elif self.architecture == "qwen-vl":
                return self._query_qwen_vl(question, image, max_new_tokens)
            else:
                return f"Error: Unsupported architecture {self.architecture}"
                
        except Exception as e:
            print(f"Error in query_with_image: {e}")
            print("Falling back to text-only approach...")
            return self.query_text_only(question, max_new_tokens)


class OptimizedEvaluationManager:
    """Manages the complete evaluation process with optimizations."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.path_manager = OptimizedPathManager(data_dir)
        self._model_cache = {}  # Model instance cache
    
    def _get_cached_model(self, model_key: str) -> OptimizedMultimodalQueryEngine:
        """Get or create cached model instance."""
        if model_key not in self._model_cache:
            self._model_cache[model_key] = OptimizedMultimodalQueryEngine(model_key)
        return self._model_cache[model_key]
    
    @lru_cache(maxsize=32)
    def _load_questions(self, category: str) -> Dict:
        """Load questions with caching."""
        question_file = self.path_manager.get_question_file(category)
        try:
            with open(question_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found for category {category}")
            return {}
    
    def _get_question_text(self, question_data: Dict, kind: str) -> str:
        """Get appropriate question text based on kind."""
        if kind == "kind1":
            return question_data["Rephrased Question(SD)"]
        else:  # kind2 and kind3
            return question_data["Rephrased Question"]
    
    def query_category(self, category: str, use_images: bool = True, kind: str = "kind3",
                      max_questions: Optional[int] = None, model_key: str = "instructblip-vicuna-7b") -> Dict[str, str]:
        """Query multimodal model for all questions in a specific category."""
        
        # Load questions
        questions_data = self._load_questions(category)
        if not questions_data:
            return {}
        
        # Get model instance
        model_engine = self._get_cached_model(model_key)
        
        # Process questions
        question_ids = list(questions_data.keys())[:max_questions] if max_questions else list(questions_data.keys())
        
        print(f"Processing {len(question_ids)} questions for {category} ({kind}) with {model_engine.display_name}...")
        
        responses = {}
        for i, question_id in enumerate(question_ids):
            print(f"  Processing question {i+1}/{len(question_ids)}: {question_id}")
            
            question_data = questions_data[question_id]
            question = self._get_question_text(question_data, kind)
            
            try:
                if use_images:
                    image_path = self.path_manager.get_image_path(category, kind, question_id)
                    response = model_engine.query_with_image(question, image_path)
                else:
                    response = model_engine.query_text_only(question)
                
                responses[question_id] = response
                print(f"  ✓ Response received for question {question_id}")
                
            except Exception as e:
                print(f"  ✗ Error processing question {question_id}: {str(e)}")
                responses[question_id] = f"Error: {str(e)}"
        
        return responses
    
    def save_responses(self, input_file_path: Path, output_file_path: Path,
                      model_name: str, responses_dict: Dict[str, str]) -> None:
        """Save model responses to JSON file efficiently."""
        
        # Read input data
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Add responses
        for question_id, response_text in responses_dict.items():
            if question_id in data:
                if "ans" not in data[question_id]:
                    data[question_id]["ans"] = {}
                data[question_id]["ans"][model_name] = {"text": response_text}
            else:
                print(f"Warning: Question ID {question_id} not found in data")
        
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Added {model_name} responses and saved to {output_file_path}")
    
    def create_answer_structure(self, input_file_path: Path, output_file_path: Path) -> Dict:
        """Create answer format structure efficiently."""
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Initialize ans fields
        for question_data in data.values():
            question_data["ans"] = {}
        
        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Created answer format structure for {len(data)} questions")
        return data
    
    def run_complete_evaluation(self, use_images: bool = True, 
                               max_questions_per_category: Optional[int] = None,
                               model_key: str = "instructblip-vicuna-7b") -> Dict:
        """Run complete MM-SafetyBench evaluation."""
        
        model_config = AVAILABLE_MODELS[model_key]
        model_display_name = model_config.display_name
        
        print("=" * 80)
        print(f"🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH {model_display_name}")
        print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
        print("=" * 80)
        print(f"🤖 Model: {model_display_name}")
        print(f"🏗️  Architecture: {model_config.architecture}")
        print(f"📊 Use images: {use_images}")
        print(f"🔢 Max questions per category per kind: {max_questions_per_category or 'ALL'}")
        
        # Display processing plan
        total_combinations = len(SAFETY_CATEGORIES) * len(KIND_MAPPINGS)
        print(f"\n🗂️  Processing {len(SAFETY_CATEGORIES)} categories × {len(KIND_MAPPINGS)} kinds = {total_combinations} total combinations:")
        for i, cat in enumerate(SAFETY_CATEGORIES, 1):
            print(f"   {i:2d}. {cat}")
        
        print(f"\n📝 Kinds to process:")
        for kind_id, (img_dir, question_text) in KIND_MAPPINGS.items():
            print(f"   • {kind_id}: {img_dir} images + '{question_text}' text")
        
        all_results = {}
        total_responses = 0
        combination_count = 0
        
        # Process each category × kind combination
        for category_idx, category in enumerate(SAFETY_CATEGORIES, 1):
            for kind_idx, (kind_id, (img_dir, question_text)) in enumerate(KIND_MAPPINGS.items(), 1):
                combination_count += 1
                
                print(f"\n" + "="*80)
                print(f"📂 COMBINATION {combination_count}/{total_combinations}")
                print(f"📁 CATEGORY {category_idx}/{len(SAFETY_CATEGORIES)}: {category}")
                print(f"🖼️  KIND {kind_idx}/{len(KIND_MAPPINGS)}: {kind_id} ({img_dir} + '{question_text}')")
                print(f"🤖 MODEL: {model_display_name}")
                print("="*80)
                
                try:
                    # Prepare file paths
                    input_file = self.path_manager.get_question_file(category)
                    temp_structure_file = self.path_manager.answer_format_dir / f"{category}_with_responses.json"
                    output_file = self.path_manager.get_output_file(model_display_name, kind_id, category)
                    
                    # Create answer structure (only once per category)
                    if kind_idx == 1:
                        print(f"📝 Creating answer structure...")
                        self.create_answer_structure(input_file, temp_structure_file)
                    
                    # Query model
                    print(f"🤖 Querying {model_display_name} with {kind_id}...")
                    responses = self.query_category(
                        category=category,
                        use_images=use_images,
                        kind=kind_id,
                        max_questions=max_questions_per_category,
                        model_key=model_key
                    )
                    
                    # Save results
                    print(f"💾 Saving results...")
                    self.save_responses(temp_structure_file, output_file, model_display_name, responses)
                    
                    # Track results
                    result_key = f"{category}_{img_dir}"
                    all_results[result_key] = {
                        'category': category,
                        'kind': kind_id,
                        'img_dir': img_dir,
                        'question_text': question_text,
                        'num_questions': len(responses),
                        'output_file': str(output_file),
                        'model': model_display_name
                    }
                    total_responses += len(responses)
                    
                    print(f"✅ {category} {kind_id} completed: {len(responses)} questions processed")
                    
                except Exception as e:
                    print(f"❌ Error processing {category} {img_dir}: {str(e)}")
                    result_key = f"{category}_{img_dir}"
                    all_results[result_key] = {
                        'category': category,
                        'kind': kind_id,
                        'img_dir': img_dir,
                        'num_questions': 0,
                        'error': str(e),
                        'model': model_display_name
                    }
        
        # Final summary
        self._print_evaluation_summary(model_display_name, total_responses, all_results)
        return all_results
    
    def _print_evaluation_summary(self, model_display_name: str, total_responses: int, all_results: Dict) -> None:
        """Print comprehensive evaluation summary."""
        
        print("\n" + "="*80)
        print("🎉 COMPLETE DATASET EVALUATION FINISHED!")
        print("🔄 ALL 3 KINDS × ALL 13 CATEGORIES PROCESSED")
        print("="*80)
        print(f"🤖 Model used: {model_display_name}")
        print(f"📊 Total questions processed: {total_responses}")
        print(f"🔢 Total combinations: {len(all_results)} (should be {len(SAFETY_CATEGORIES)} × {len(KIND_MAPPINGS)} = {len(SAFETY_CATEGORIES) * len(KIND_MAPPINGS)})")
        print(f"✅ Successful combinations: {len([r for r in all_results.values() if 'error' not in r])}")
        print(f"❌ Failed combinations: {len([r for r in all_results.values() if 'error' in r])}")
        
        print("\n📋 DETAILED RESULTS BY CATEGORY:")
        for category in SAFETY_CATEGORIES:
            print(f"\n📁 {category}:")
            for kind_id, (img_dir, _) in KIND_MAPPINGS.items():
                result_key = f"{category}_{img_dir}"
                if result_key in all_results:
                    result = all_results[result_key]
                    if 'error' in result:
                        print(f"   ❌ {img_dir}: ERROR - {result['error']}")
                    else:
                        print(f"   ✅ {img_dir}: {result['num_questions']} questions → {result['output_file']}")
                else:
                    print(f"   ⚠️  {img_dir}: NOT PROCESSED")
        
        print(f"\n🎯 All results saved in: {self.path_manager.results_dir}")
        print(f"📁 Directory structure: {{model_name}}/{{image_type}}/{{category}}_with_responses.json")
        print("="*80)


def print_available_models() -> None:
    """Print available multimodal models for selection."""
    print("\n🤖 Available Multimodal Models:")
    print("="*80)
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key:25} → {config.display_name}")
        print(f"  {' '*25}   Architecture: {config.architecture}")
        print(f"  {' '*25}   {config.description}")
        print()


def interactive_model_selection() -> Optional[str]:
    """Interactive model selection for users."""
    print_available_models()
    
    while True:
        choice = input("🔤 Enter model key (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            return None
        
        if choice in AVAILABLE_MODELS:
            config = AVAILABLE_MODELS[choice]
            print(f"\n✅ Selected: {config.display_name}")
            print(f"🏗️  Architecture: {config.architecture}")
            print(f"📝 Description: {config.description}")
            return choice
        else:
            print(f"❌ Invalid choice: '{choice}'")
            print(f"✅ Valid options: {', '.join(AVAILABLE_MODELS.keys())}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with improved help."""
    parser = argparse.ArgumentParser(
        description="Optimized MM-SafetyBench evaluation with multimodal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  cogvlm-chat-17b           → CogVLM-Chat(17B)           - THUDM, high performance
  cogvlm2-llama3-chat-19b   → CogVLM2-Llama3-Chat(19B)  - THUDM, latest version  
  instructblip-vicuna-7b    → InstructBLIP-Vicuna(7B)   - Salesforce, instruction-following
  instructblip-vicuna-13b   → InstructBLIP-Vicuna(13B)  - Salesforce, larger variant
  qwen-vl-chat             → Qwen-VL-Chat              - Alibaba, multilingual
  qwen-vl                  → Qwen-VL                   - Alibaba, base model

Setup Requirements:
  1. Install transformers: pip install transformers torch torchvision
  2. Some models require: trust_remote_code=True
  3. Ensure CUDA is available for GPU acceleration
  4. Models will be downloaded automatically from HuggingFace

Examples:
  python add_multimodal_responses_optimized.py --model instructblip-vicuna-7b --max-questions 5
  python add_multimodal_responses_optimized.py --model cogvlm2-llama3-chat-19b --text-only
  python add_multimodal_responses_optimized.py --interactive
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=list(AVAILABLE_MODELS.keys()),
        default="instructblip-vicuna-7b",
        help="Multimodal model to use (default: instructblip-vicuna-7b)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=5,
        help="Maximum questions per category per kind (default: 5, use 0 for all)"
    )
    
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Use text-only queries (no images)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true", 
        help="Interactive model selection"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Path to MM-SafetyBench data directory (default: {DEFAULT_DATA_DIR})"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Handle special flags
    if args.list_models:
        print_available_models()
        return
    
    # Determine model to use
    if args.interactive:
        model_key = interactive_model_selection()
        if model_key is None:
            return
    else:
        model_key = args.model
    
    # Determine max questions (0 means all)
    max_questions = args.max_questions if args.max_questions > 0 else None
    
    # Initialize evaluation manager
    eval_manager = OptimizedEvaluationManager(args.data_dir)
    
    # Run evaluation
    print("\n" + "="*80)
    print("🚀 STARTING OPTIMIZED MM-SAFETYBENCH EVALUATION WITH MULTIMODAL MODELS")
    print("="*80)
    print(f"🤖 Model: {AVAILABLE_MODELS[model_key].display_name}")
    print(f"🏗️  Architecture: {AVAILABLE_MODELS[model_key].architecture}")
    print(f"🖼️  Images: {'Disabled (text-only)' if args.text_only else 'Enabled'}")
    print(f"🔢 Questions per category: {max_questions or 'ALL'}")
    print("="*80)
    
    # Run complete evaluation
    eval_manager.run_complete_evaluation(
        use_images=not args.text_only,
        max_questions_per_category=max_questions,
        model_key=model_key
    )
    
    print("\n" + "="*80)
    print("🔧 QUICK REFERENCE")
    print("="*80)
    print("📋 Available models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  --model {key:25} → {config.display_name}")
    print("\n🎛️  Command-line options:")
    print("  --model <model_key>        Choose multimodal model")
    print("  --max-questions <N>        Process N questions per category (0 = all)")
    print("  --text-only                Skip images, use text-only queries")
    print("  --interactive              Interactive model selection")
    print("  --list-models              Show available models")
    print("  --data-dir <path>          Path to MM-SafetyBench data directory")
    print("\n📝 Examples:")
    print("  python add_multimodal_responses_optimized.py --model cogvlm2-llama3-chat-19b --max-questions 10")
    print("  python add_multimodal_responses_optimized.py --interactive")
    print("="*80)


if __name__ == "__main__":
    main() 