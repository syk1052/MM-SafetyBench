#!/usr/bin/env python3
"""
Optimized MM-SafetyBench MiniGPT Evaluation Script

This optimized version maintains all original functionality while improving:
- Performance and memory efficiency
- Code organization and maintainability  
- Error handling and resource management
- Constants and configuration management
"""

import json
import os
import torch
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
DEFAULT_MINIGPT_REPO = "../.cache/MiniGPT-4"
DEFAULT_DATA_DIR = "../data/MM-SafetyBench"
DEFAULT_MAX_NEW_TOKENS = 500
DEFAULT_MAX_LENGTH = 2000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_NUM_BEAMS = 1

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
    """Configuration for MiniGPT models."""
    display_name: str
    description: str
    backbone: str
    architecture: str
    config_path: str

# Available MiniGPT models (using dataclass for better organization)
AVAILABLE_MODELS = {
    "minigpt4-vicuna-7b": ModelConfig(
        display_name="MiniGPT4-Vicuna(7B)",
        description="MiniGPT-4 with Vicuna 7B backbone (faster, less memory)",
        backbone="vicuna-7b",
        architecture="minigpt4",
        config_path="eval_configs/minigpt4_eval_7b.yaml"
    ),
    "minigpt4-vicuna-13b": ModelConfig(
        display_name="MiniGPT4-Vicuna(13B)",
        description="MiniGPT-4 with Vicuna 13B backbone (slower, more memory, potentially better)",
        backbone="vicuna-13b",
        architecture="minigpt4",
        config_path="eval_configs/minigpt4_eval_13b.yaml"
    ),
    "minigpt4-llama2": ModelConfig(
        display_name="MiniGPT4-Llama2(7B)",
        description="MiniGPT-4 with Llama2 7B backbone (official Llama2 support)",
        backbone="llama2-7b",
        architecture="minigpt4",
        config_path="eval_configs/minigpt4_llama2_eval.yaml"
    ),
    "minigptv2-llama2": ModelConfig(
        display_name="MiniGPT-v2-Llama2(7B)",
        description="MiniGPT-v2 with Llama2 7B backbone (improved version with better performance)",
        backbone="llama2-7b",
        architecture="minigptv2",
        config_path="eval_configs/minigptv2_eval.yaml"
    ),
    "minigpt5-7b": ModelConfig(
        display_name="MiniGPT-5(7B)",
        description="MiniGPT-5 with interleaved vision-and-language generation (experimental)",
        backbone="llama2-7b",
        architecture="minigpt5",
        config_path="eval_configs/minigpt5_eval.yaml"
    ),
    "minigpt5-v2-7b": ModelConfig(
        display_name="MiniGPT-5-V2(7B)",
        description="MiniGPT-5 V2 with improved generative capabilities (experimental)",
        backbone="llama2-7b",
        architecture="minigpt5v2",
        config_path="eval_configs/minigpt5v2_eval.yaml"
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
    def extract_response_text(llm_message) -> str:
        """Extract text from MiniGPT response (handles tuple with text and token arrays)."""
        if isinstance(llm_message, tuple) and len(llm_message) > 0:
            return str(llm_message[0])
        return str(llm_message)


class OptimizedMiniGPTQueryEngine:
    """Optimized MiniGPT query engine with better resource management."""
    
    def __init__(self, model_key: str = "minigpt4-vicuna-7b", 
                 minigpt_repo_path: str = DEFAULT_MINIGPT_REPO):
        """Initialize MiniGPT model for querying."""
        self.model_config = self._get_model_config(model_key)
        self.model_key = model_key
        self.display_name = self.model_config.display_name
        self.minigpt_repo_path = Path(minigpt_repo_path)
        
        # Model components (lazy loading)
        self.model = None
        self.vis_processor = None
        self.chat = None
        self.conv_template = None
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Selected model: {self.display_name}")
        
        # Add MiniGPT repository to path
        if str(self.minigpt_repo_path) not in sys.path:
            sys.path.append(str(self.minigpt_repo_path))
    
    @staticmethod
    def _get_model_config(model_key: str) -> ModelConfig:
        """Get model configuration with validation."""
        if model_key not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")
        return AVAILABLE_MODELS[model_key]
    
    def _apply_patches(self) -> None:
        """Apply all necessary patches for compatibility."""
        # Apply text-only patch
        print("🔧 Applying text-only patch...")
        try:
            from fix_minigpt_textonly import patch_minigpt_textonly
            if not patch_minigpt_textonly():
                print("⚠️ Warning: Text-only patch failed, text-only queries may not work")
        except ImportError:
            print("⚠️ Warning: Text-only patch not found, text-only queries may not work")
        
        # Apply transformers compatibility patches
        print("🔧 Applying transformers compatibility patches...")
        try:
            from fix_transformers_compatibility import (
                patch_transformers_compatibility, 
                patch_generation_utils, 
                patch_internal_generation_methods
            )
            if patch_transformers_compatibility():
                print("✅ Transformers compatibility patch applied")
            patch_generation_utils()
            patch_internal_generation_methods()
        except ImportError:
            print("⚠️ Warning: Transformers compatibility patches not found")
    
    def _get_config_path(self, config_path: Optional[str] = None) -> Path:
        """Get and validate configuration path."""
        if config_path is None:
            config_path = self.minigpt_repo_path / self.model_config.config_path
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            config_path = self._handle_missing_config(config_path)
        
        return config_path
    
    def _handle_missing_config(self, original_config_path: Path) -> Path:
        """Handle missing configuration files with smart fallback."""
        print(f"Attempting to resolve missing config: {original_config_path}")
        
        # Try common alternative locations
        possible_locations = [
            Path.home() / ".cache/MiniGPT-4/eval_configs" / original_config_path.name,
            self.minigpt_repo_path / "minigpt4/configs/models" / original_config_path.name.replace("_eval", ""),
        ]
        
        for location in possible_locations:
            if location.exists():
                print(f"Found alternative config at: {location}")
                return location
        
        # Smart fallback based on model type
        if "7b" in self.model_key.lower():
            fallback_config = self.minigpt_repo_path / "eval_configs/minigpt4_eval_7b.yaml"
        elif "13b" in self.model_key.lower():
            fallback_config = self.minigpt_repo_path / "eval_configs/minigpt4_eval_13b.yaml"
        else:
            fallback_config = self.minigpt_repo_path / "eval_configs/minigpt4_eval.yaml"
        
        if fallback_config.exists():
            print(f"Using fallback config: {fallback_config}")
            return fallback_config
        
        raise FileNotFoundError(f"Could not find or create config file: {original_config_path}")
    
    def load_model(self, config_path: Optional[str] = None, 
                   checkpoint_path: Optional[str] = None) -> None:
        """Load the MiniGPT model with optimized initialization."""
        try:
            print(f"Loading MiniGPT model: {self.display_name}")
            print(f"Using device: {self.device}")
            
            # Import MiniGPT modules
            from minigpt4.common.config import Config
            from minigpt4.common.registry import registry
            from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
            
            # Apply patches
            self._apply_patches()
            
            # Get configuration path
            config_path = self._get_config_path(config_path)
            
            # Load configuration
            class Args:
                def __init__(self, cfg_path):
                    self.cfg_path = str(cfg_path)
                    self.options = []
            
            cfg = Config(Args(config_path))
            
            # Handle 13B model specific configuration
            if "13b" in self.model_key:
                if hasattr(cfg.model_cfg, 'model_type'):
                    cfg.model_cfg.model_type = "pretrain_vicuna13b"
                print(f"Using 13B model configuration: {cfg.model_cfg.model_type}")
            
            # Update checkpoint path if provided
            if checkpoint_path:
                cfg.model_cfg.ckpt = checkpoint_path
            
            # Build model
            model_cls = registry.get_model_class(cfg.model_cfg.arch)
            self.model = model_cls.from_config(cfg.model_cfg).to(self.device)
            
            # Build vision processor
            vis_processor_cls = registry.get_processor_class(
                cfg.datasets_cfg.cc_sbu_align.vis_processor.train.name
            )
            self.vis_processor = vis_processor_cls.from_config(
                cfg.datasets_cfg.cc_sbu_align.vis_processor.train
            )
            
            # Set conversation template based on model type
            self.conv_template = CONV_VISION_Vicuna0 if "vicuna" in self.model_key else CONV_VISION_LLama2
            
            # Initialize chat
            self.chat = Chat(self.model, self.vis_processor, device=self.device)
            
            print("Model loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Error importing MiniGPT modules: {e}")
            print("💡 Make sure you have cloned and set up the MiniGPT-4 repository:")
            print("   git clone https://github.com/Vision-CAIR/MiniGPT-4.git")
            print("   cd MiniGPT-4")
            print("   pip install -e .")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _prepare_conversation_state(self) -> object:
        """Prepare conversation state with proper role initialization."""
        chat_state = self.conv_template.copy()
        
        # Check if roles are properly initialized
        if not hasattr(chat_state, 'roles') or len(chat_state.roles) < 2:
            print("Warning: Conversation template roles not properly initialized")
            # Set default roles based on model type
            if "vicuna" in self.model_key:
                chat_state.roles = ["USER", "ASSISTANT"]
            else:
                chat_state.roles = ["Human", "Assistant"]
        
        return chat_state
    
    def query_text_only(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query MiniGPT with text only (no image)."""
        if self.model is None:
            self.load_model()
        
        try:
            # Prepare conversation state
            chat_state = self._prepare_conversation_state()
            
            # Safely access roles
            user_role = chat_state.roles[0] if len(chat_state.roles) > 0 else "USER"
            assistant_role = chat_state.roles[1] if len(chat_state.roles) > 1 else "ASSISTANT"
            
            # Set up conversation
            chat_state.append_message(user_role, question)
            chat_state.append_message(assistant_role, None)
            
            # Generate response
            with torch.no_grad():
                llm_message = self.chat.answer(
                    conv=chat_state,
                    img_list=[],  # No images
                    num_beams=DEFAULT_NUM_BEAMS,
                    temperature=DEFAULT_TEMPERATURE,
                    max_new_tokens=max_new_tokens,
                    max_length=DEFAULT_MAX_LENGTH
                )
            
            return ModelResponseProcessor.extract_response_text(llm_message)
            
        except Exception as e:
            print(f"Error in query_text_only: {e}")
            return f"Error: {str(e)}"
    
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image with error handling."""
        try:
            image = Image.open(image_path)
            print(f"Loaded image: {image_path}")
            return image
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            return None
    
    def query_with_image(self, question: str, image_path: Path, 
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query MiniGPT with image and text."""
        if self.model is None:
            self.load_model()
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            print("Image not found, falling back to text-only...")
            return self.query_text_only(question, max_new_tokens)
        
        try:
            # Initialize conversation state
            chat_state = self.conv_template.copy()
            img_list = []
            
            # Upload and process image
            self.chat.upload_img(image, chat_state, img_list)
            
            # Ask question
            self.chat.ask(question, chat_state)
            
            # Generate response
            llm_message = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=DEFAULT_NUM_BEAMS,
                temperature=DEFAULT_TEMPERATURE,
                max_new_tokens=max_new_tokens,
                max_length=DEFAULT_MAX_LENGTH
            )
            
            return ModelResponseProcessor.extract_response_text(llm_message)
            
        except Exception as e:
            print(f"Error in query_with_image: {e}")
            print("Falling back to text-only approach...")
            return self.query_text_only(question, max_new_tokens)


class OptimizedEvaluationManager:
    """Manages the complete evaluation process with optimizations."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.path_manager = OptimizedPathManager(data_dir)
        self._model_cache = {}  # Model instance cache
    
    def _get_cached_model(self, model_key: str, minigpt_repo_path: str) -> OptimizedMiniGPTQueryEngine:
        """Get or create cached model instance."""
        cache_key = f"{model_key}_{minigpt_repo_path}"
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = OptimizedMiniGPTQueryEngine(model_key, minigpt_repo_path)
        return self._model_cache[cache_key]
    
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
                      max_questions: Optional[int] = None, model_key: str = "minigpt4-vicuna-7b",
                      minigpt_repo_path: str = DEFAULT_MINIGPT_REPO) -> Dict[str, str]:
        """Query MiniGPT for all questions in a specific category."""
        
        # Load questions
        questions_data = self._load_questions(category)
        if not questions_data:
            return {}
        
        # Get model instance
        model_engine = self._get_cached_model(model_key, minigpt_repo_path)
        
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
                               model_key: str = "minigpt4-vicuna-7b",
                               minigpt_repo_path: str = DEFAULT_MINIGPT_REPO) -> Dict:
        """Run complete MM-SafetyBench evaluation."""
        
        model_config = AVAILABLE_MODELS[model_key]
        model_display_name = model_config.display_name
        
        print("=" * 80)
        print(f"🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH {model_display_name}")
        print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
        print("=" * 80)
        print(f"🤖 Model: {model_display_name}")
        print(f"📂 MiniGPT Repo: {minigpt_repo_path}")
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
                        model_key=model_key,
                        minigpt_repo_path=minigpt_repo_path
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
    """Print available MiniGPT models for selection."""
    print("\n🤖 Available MiniGPT Models:")
    print("="*60)
    for key, config in AVAILABLE_MODELS.items():
        print(f"  {key:20} → {config.display_name}")
        print(f"  {' '*20}   {config.description}")
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
            print(f"📝 Description: {config.description}")
            return choice
        else:
            print(f"❌ Invalid choice: '{choice}'")
            print(f"✅ Valid options: {', '.join(AVAILABLE_MODELS.keys())}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with improved help."""
    parser = argparse.ArgumentParser(
        description="Optimized MM-SafetyBench evaluation with MiniGPT models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  minigpt4-vicuna-7b    → MiniGPT4-Vicuna(7B)    - Faster, less memory
  minigpt4-vicuna-13b   → MiniGPT4-Vicuna(13B)   - Slower, more memory, potentially better
  minigpt4-llama2       → MiniGPT4-Llama2(7B)    - Official Llama2 support
  minigptv2-llama2      → MiniGPT-v2-Llama2(7B)  - Improved version
  minigpt5-7b           → MiniGPT-5(7B)          - Interleaved generation (experimental)
  minigpt5-v2-7b        → MiniGPT-5-V2(7B)       - Improved generative (experimental)

Setup Requirements:
  1. Clone MiniGPT-4 repository: git clone https://github.com/Vision-CAIR/MiniGPT-4.git
  2. Install dependencies: pip install -e .
  3. Download model checkpoints and configure paths
  4. Update --minigpt-repo-path if needed

Examples:
  python add_minigpt_responses_optimized.py --model minigpt4-vicuna-7b --max-questions 5
  python add_minigpt_responses_optimized.py --model minigptv2-llama2 --text-only
  python add_minigpt_responses_optimized.py --interactive
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=list(AVAILABLE_MODELS.keys()),
        default="minigpt4-vicuna-7b",
        help="MiniGPT model to use (default: minigpt4-vicuna-7b)"
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
        "--minigpt-repo-path",
        type=str,
        default=DEFAULT_MINIGPT_REPO,
        help=f"Path to the MiniGPT-4 repository (default: {DEFAULT_MINIGPT_REPO})"
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
    
    # Check if MiniGPT repository exists
    if not Path(args.minigpt_repo_path).exists():
        print(f"❌ MiniGPT repository not found at: {args.minigpt_repo_path}")
        print("💡 Please clone the MiniGPT-4 repository:")
        print("   git clone https://github.com/Vision-CAIR/MiniGPT-4.git")
        print("   OR update --minigpt-repo-path to point to your MiniGPT-4 directory")
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
    print("🚀 STARTING OPTIMIZED MM-SAFETYBENCH EVALUATION WITH MINIGPT MODELS")
    print("="*80)
    print(f"🤖 Model: {AVAILABLE_MODELS[model_key].display_name}")
    print(f"📂 MiniGPT Repo: {args.minigpt_repo_path}")
    print(f"🖼️  Images: {'Disabled (text-only)' if args.text_only else 'Enabled'}")
    print(f"🔢 Questions per category: {max_questions or 'ALL'}")
    print("="*80)
    
    # Run complete evaluation
    eval_manager.run_complete_evaluation(
        use_images=not args.text_only,
        max_questions_per_category=max_questions,
        model_key=model_key,
        minigpt_repo_path=args.minigpt_repo_path
    )
    
    print("\n" + "="*80)
    print("🔧 QUICK REFERENCE")
    print("="*80)
    print("📋 Available models:")
    for key, config in AVAILABLE_MODELS.items():
        print(f"  --model {key:20} → {config.display_name}")
    print("\n🎛️  Command-line options:")
    print("  --model <model_key>        Choose MiniGPT model")
    print("  --max-questions <N>        Process N questions per category (0 = all)")
    print("  --text-only                Skip images, use text-only queries")
    print("  --interactive              Interactive model selection")
    print("  --list-models              Show available models")
    print("  --minigpt-repo-path <path> Path to MiniGPT-4 repository")
    print("  --data-dir <path>          Path to MM-SafetyBench data directory")
    print("\n📝 Examples:")
    print("  python add_minigpt_responses_optimized.py --model minigpt4-vicuna-13b --max-questions 10")
    print("  python add_minigpt_responses_optimized.py --interactive")
    print("="*80)


if __name__ == "__main__":
    main() 