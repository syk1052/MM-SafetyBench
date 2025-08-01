#!/usr/bin/env python3
"""
Optimized MM-SafetyBench LLaVA Evaluation Script

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
from transformers import LlavaProcessor, LlavaForConditionalGeneration

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
    """Configuration for LLaVA models."""
    display_name: str
    description: str
    hf_name: str
    architecture: str

# Available LLaVA models (using dataclass for better organization)
AVAILABLE_MODELS = {
    "llava-1.5-7b": ModelConfig(
        display_name="LLaVA-1.5(7B)",
        description="LLaVA 1.5 with 7B parameters (faster, less memory)",
        hf_name="llava-hf/llava-1.5-7b-hf",
        architecture="llava1.5"
    ),
    "llava-1.5-13b": ModelConfig(
        display_name="LLaVA-1.5(13B)",
        description="LLaVA 1.5 with 13B parameters (slower, more memory, potentially better performance)",
        hf_name="llava-hf/llava-1.5-13b-hf",
        architecture="llava1.5"
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
        """Extract text from LLaVA response."""
        return str(response)


class OptimizedLLaVAQueryEngine:
    """Optimized LLaVA query engine with better resource management."""
    
    def __init__(self, model_key: str = "llava-1.5-7b"):
        """Initialize LLaVA model for querying."""
        self.model_config = self._get_model_config(model_key)
        self.model_key = model_key
        self.display_name = self.model_config.display_name
        self.hf_model_name = self.model_config.hf_name
        
        # Model components (lazy loading)
        self.processor = None
        self.model = None
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Selected model: {self.display_name}")
    
    @staticmethod
    def _get_model_config(model_key: str) -> ModelConfig:
        """Get model configuration with validation."""
        if model_key not in AVAILABLE_MODELS:
            available = list(AVAILABLE_MODELS.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")
        return AVAILABLE_MODELS[model_key]
    
    def load_model(self) -> None:
        """Load the LLaVA model and processor with optimized initialization."""
        try:
            print(f"Loading LLaVA model: {self.display_name}")
            print(f"HuggingFace model: {self.hf_model_name}")
            print(f"Using device: {self.device}")
            
            self.processor = LlavaProcessor.from_pretrained(self.hf_model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load image with error handling."""
        try:
            image = Image.open(image_path)
            print(f"Loaded image: {image_path}")
            return image
        except Exception as e:
            print(f"Could not load image {image_path}: {e}")
            return None
    
    def query_text_only(self, question: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query LLaVA with text only (no image)."""
        if self.model is None:
            self.load_model()
        
        prompt = f"USER: {question}\nASSISTANT:"
        
        try:
            inputs = self.processor(text=prompt, images=None, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=DEFAULT_TEMPERATURE,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode and extract the response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in query_text_only: {e}")
            return f"Error: {str(e)}"
    
    def query_with_image(self, question: str, image_path: Path, 
                        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
        """Query LLaVA with image and text."""
        if self.model is None:
            self.load_model()
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            print("Image not found, falling back to text-only...")
            return self.query_text_only(question, max_new_tokens)
        
        # Use the proper LLaVA format with image token
        prompt = f"<image>\nUSER: {question}\nASSISTANT:"
        
        try:
            # Process the image and text
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            
            # Move inputs to device, handling each tensor individually
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=DEFAULT_TEMPERATURE,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode and extract the response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return ModelResponseProcessor.extract_response_text(response)
            
        except Exception as e:
            print(f"Error in query_with_image: {e}")
            print("Falling back to text-only approach...")
            return self.query_text_only(question, max_new_tokens)


class OptimizedEvaluationManager:
    """Manages the complete evaluation process with optimizations."""
    
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.path_manager = OptimizedPathManager(data_dir)
        self._model_cache = {}  # Model instance cache
    
    def _get_cached_model(self, model_key: str) -> OptimizedLLaVAQueryEngine:
        """Get or create cached model instance."""
        if model_key not in self._model_cache:
            self._model_cache[model_key] = OptimizedLLaVAQueryEngine(model_key)
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
                      max_questions: Optional[int] = None, model_key: str = "llava-1.5-7b") -> Dict[str, str]:
        """Query LLaVA for all questions in a specific category."""
        
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
                               model_key: str = "llava-1.5-7b") -> Dict:
        """Run complete MM-SafetyBench evaluation."""
        
        model_config = AVAILABLE_MODELS[model_key]
        model_display_name = model_config.display_name
        
        print("=" * 80)
        print(f"🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH {model_display_name}")
        print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
        print("=" * 80)
        print(f"🤖 Model: {model_display_name}")
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
    """Print available LLaVA models for selection."""
    print("\n🤖 Available LLaVA Models:")
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
        description="Optimized MM-SafetyBench evaluation with LLaVA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  llava-1.5-7b          → LLaVA-1.5(7B)         - Faster, less memory
  llava-1.5-13b         → LLaVA-1.5(13B)        - Slower, more memory, potentially better

Setup Requirements:
  1. Install transformers: pip install transformers torch torchvision
  2. Ensure CUDA is available for GPU acceleration
  3. Models will be downloaded automatically from HuggingFace

Examples:
  python add_llava_responses_optimized.py --model llava-1.5-7b --max-questions 5
  python add_llava_responses_optimized.py --model llava-1.5-13b --text-only
  python add_llava_responses_optimized.py --interactive
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=list(AVAILABLE_MODELS.keys()),
        default="llava-1.5-7b",
        help="LLaVA model to use (default: llava-1.5-7b)"
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
    print("🚀 STARTING OPTIMIZED MM-SAFETYBENCH EVALUATION WITH LLAVA MODELS")
    print("="*80)
    print(f"🤖 Model: {AVAILABLE_MODELS[model_key].display_name}")
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
        print(f"  --model {key:20} → {config.display_name}")
    print("\n🎛️  Command-line options:")
    print("  --model <model_key>        Choose LLaVA model")
    print("  --max-questions <N>        Process N questions per category (0 = all)")
    print("  --text-only                Skip images, use text-only queries")
    print("  --interactive              Interactive model selection")
    print("  --list-models              Show available models")
    print("  --data-dir <path>          Path to MM-SafetyBench data directory")
    print("\n📝 Examples:")
    print("  python add_llava_responses_optimized.py --model llava-1.5-13b --max-questions 10")
    print("  python add_llava_responses_optimized.py --interactive")
    print("="*80)


if __name__ == "__main__":
    main() 