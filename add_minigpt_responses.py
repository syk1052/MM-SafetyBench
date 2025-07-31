import json
import os
import torch
import argparse
from PIL import Image
import sys
import warnings
warnings.filterwarnings("ignore")

# Note: MiniGPT models require custom setup and do not use standard HuggingFace transformers
# This implementation provides a framework but requires the MiniGPT-4/v2 repository setup

# Available MiniGPT models
AVAILABLE_MODELS = {
    "minigpt4-vicuna-7b": {
        "display_name": "MiniGPT4-Vicuna(7B)",
        "description": "MiniGPT-4 with Vicuna 7B backbone (faster, less memory)",
        "backbone": "vicuna-7b",
        "architecture": "minigpt4",
        "config_path": "eval_configs/minigpt4_eval_7b.yaml"  # Updated for 7B model
    },
    "minigpt4-vicuna-13b": {
        "display_name": "MiniGPT4-Vicuna(13B)",
        "description": "MiniGPT-4 with Vicuna 13B backbone (slower, more memory, potentially better)",
        "backbone": "vicuna-13b", 
        "architecture": "minigpt4",
        "config_path": "eval_configs/minigpt4_eval_13b.yaml"
    },
    "minigpt4-llama2": {
        "display_name": "MiniGPT4-Llama2(7B)",
        "description": "MiniGPT-4 with Llama2 7B backbone (official Llama2 support)",
        "backbone": "llama2-7b",
        "architecture": "minigpt4",
        "config_path": "eval_configs/minigpt4_llama2_eval.yaml"
    },
    "minigptv2-llama2": {
        "display_name": "MiniGPT-v2-Llama2(7B)",
        "description": "MiniGPT-v2 with Llama2 7B backbone (improved version with better performance)",
        "backbone": "llama2-7b",
        "architecture": "minigptv2",
        "config_path": "eval_configs/minigptv2_eval.yaml"
    },
    "minigpt5-7b": {
        "display_name": "MiniGPT-5(7B)",
        "description": "MiniGPT-5 with interleaved vision-and-language generation (experimental)",
        "backbone": "llama2-7b",
        "architecture": "minigpt5",
        "config_path": "eval_configs/minigpt5_eval.yaml"  # hypothetical
    },
    "minigpt5-v2-7b": {
        "display_name": "MiniGPT-5-V2(7B)",
        "description": "MiniGPT-5 V2 with improved generative capabilities (experimental)",
        "backbone": "llama2-7b",
        "architecture": "minigpt5v2",
        "config_path": "eval_configs/minigpt5v2_eval.yaml"  # hypothetical
    }
}

def print_available_models():
    """Print available MiniGPT models for selection."""
    print("\n🤖 Available MiniGPT Models:")
    print("="*60)
    for key, info in AVAILABLE_MODELS.items():
        print(f"  {key:20} → {info['display_name']}")
        print(f"  {' '*20}   {info['description']}")
        print()

def get_model_info(model_key):
    """Get model information from the model key."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_key]

class MiniGPTQueryEngine:
    def __init__(self, model_key="minigpt4-vicuna-7b", minigpt_repo_path="../.cache/MiniGPT-4"):
        """
        Initialize MiniGPT model for querying.
        
        Args:
            model_key (str): Model key from AVAILABLE_MODELS
            minigpt_repo_path (str): Path to the MiniGPT-4 repository
        """
        self.model_info = get_model_info(model_key)
        self.model_key = model_key
        self.display_name = self.model_info["display_name"]
        self.minigpt_repo_path = minigpt_repo_path
        self.model = None
        self.vis_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Selected model: {self.display_name}")
        
        # Add MiniGPT repository to path
        if minigpt_repo_path not in sys.path:
            sys.path.append(minigpt_repo_path)
            
    def load_model(self, config_path=None, checkpoint_path=None):
        """
        Load the MiniGPT model.
        
        Args:
            config_path (str): Path to config file (optional, uses default if None)
            checkpoint_path (str): Path to checkpoint file (optional)
        """
        try:
            print(f"Loading MiniGPT model: {self.display_name}")
            print(f"Using device: {self.device}")
            
            # Import MiniGPT modules (requires MiniGPT-4 repo in path)
            from minigpt4.common.config import Config
            from minigpt4.common.registry import registry
            from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2
            
            # Apply text-only patch for empty img_list support
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
                from fix_transformers_compatibility import patch_transformers_compatibility, patch_generation_utils, patch_internal_generation_methods
                if patch_transformers_compatibility():
                    print("✅ Transformers compatibility patch applied")
                patch_generation_utils()
                patch_internal_generation_methods()
            except ImportError:
                print("⚠️ Warning: Transformers compatibility patches not found")
            
            # Set default config path if not provided
            if config_path is None:
                config_path = os.path.join(self.minigpt_repo_path, self.model_info["config_path"])
            
            # Check if config file exists, if not, try to create or use alternative
            if not os.path.exists(config_path):
                print(f"Config file not found: {config_path}")
                config_path = self._handle_missing_config(config_path)
                
            # Load configuration
            # Create args object that Config expects
            class Args:
                def __init__(self, cfg_path):
                    self.cfg_path = cfg_path
                    self.options = []
            
            args = Args(config_path)
            cfg = Config(args)
            
            # Handle 13B model specific configuration
            if "13b" in self.model_key:
                # Force the correct model type for 13B models
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
            vis_processor_cls = registry.get_processor_class(cfg.datasets_cfg.cc_sbu_align.vis_processor.train.name)
            self.vis_processor = vis_processor_cls.from_config(cfg.datasets_cfg.cc_sbu_align.vis_processor.train)
            
            # Set conversation template based on model type
            if "vicuna" in self.model_key:
                self.conv_template = CONV_VISION_Vicuna0
            else:  # llama2
                self.conv_template = CONV_VISION_LLama2
                
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

    def _handle_missing_config(self, original_config_path):
        """
        Handle missing configuration files by creating them or finding alternatives.
        
        Args:
            original_config_path (str): The original config path that was not found
            
        Returns:
            str: Path to a valid configuration file
        """
        print(f"Attempting to resolve missing config: {original_config_path}")
        
        # Extract the config filename
        config_filename = os.path.basename(original_config_path)
        config_dir = os.path.dirname(original_config_path)
        
        # Try to find the config in common locations
        possible_locations = [
            os.path.join("/home/.cache/MiniGPT-4/eval_configs", config_filename),
            os.path.join("/home/.cache/MiniGPT-4/minigpt4/configs/models", config_filename.replace("_eval", "")),
            os.path.join(self.minigpt_repo_path, "minigpt4/configs/models", config_filename.replace("_eval", "")),
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                print(f"Found alternative config at: {location}")
                return location
        
        # If 13B config is missing, create it based on the 7B config
        if "13b" in config_filename:
            print("Creating 13B configuration file...")
            return self._create_13b_config(original_config_path)
        
        # If nothing found, try using the 7B config as fallback
        fallback_config = original_config_path.replace("13b", "7b").replace("_13b", "_7b")
        if os.path.exists(fallback_config):
            print(f"Using fallback config: {fallback_config}")
            return fallback_config
        
        # Last resort - use a compatible default config based on model type
        if "7b" in self.model_key.lower():
            default_config = os.path.join(self.minigpt_repo_path, "eval_configs/minigpt4_eval_7b.yaml")
        elif "13b" in self.model_key.lower():
            default_config = os.path.join(self.minigpt_repo_path, "eval_configs/minigpt4_eval_13b.yaml")
        else:
            default_config = os.path.join(self.minigpt_repo_path, "eval_configs/minigpt4_eval.yaml")
        
        if os.path.exists(default_config):
            print(f"Using fallback config: {default_config}")
            return default_config
        
        raise FileNotFoundError(f"Could not find or create config file: {original_config_path}")
    
    def _create_13b_config(self, target_path):
        """
        Create a 13B configuration file based on the 7B template.
        
        Args:
            target_path (str): Where to create the 13B config file
            
        Returns:
            str: Path to the created configuration file
        """
        # Read the 7B config as template
        template_path = target_path.replace("13b", "7b").replace("_13b", "_7b")
        
        if not os.path.exists(template_path):
            template_path = os.path.join(self.minigpt_repo_path, "eval_configs/minigpt4_eval.yaml")
        
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Could not find template config: {template_path}")
        
        # Read template and modify for 13B
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Update model type for 13B
        content = content.replace("model_type: pretrain_vicuna0", "model_type: pretrain_vicuna13b")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Write the new config
        with open(target_path, 'w') as f:
            f.write(content)
        
        print(f"Created 13B config file: {target_path}")
        return target_path
    
    def query_text_only(self, question, max_new_tokens=500):
        """
        Query MiniGPT with text only (no image).
        Note: MiniGPT models are primarily designed for vision-language tasks.
        """
        if self.model is None:
            self.load_model()
        
        try:
            # For text-only, we'll create a minimal conversation
            chat_state = self.conv_template.copy()
            
            # Check if roles are properly initialized
            if not hasattr(chat_state, 'roles') or len(chat_state.roles) < 2:
                print("Warning: Conversation template roles not properly initialized")
                # Try to set default roles based on model type
                if "vicuna" in self.model_key:
                    chat_state.roles = ["USER", "ASSISTANT"]
                else:
                    chat_state.roles = ["Human", "Assistant"]
            
            # Safely access roles
            user_role = chat_state.roles[0] if len(chat_state.roles) > 0 else "USER"
            assistant_role = chat_state.roles[1] if len(chat_state.roles) > 1 else "ASSISTANT"
            
            chat_state.append_message(user_role, question)
            chat_state.append_message(assistant_role, None)
            
            # Get prompt
            prompt = chat_state.get_prompt()
            
            # Generate response (without image)
            with torch.no_grad():
                llm_message = self.chat.answer(
                    conv=chat_state,
                    img_list=[],  # No images
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=max_new_tokens,
                    max_length=2000
                )
            
            # Extract text from MiniGPT response (it returns tuple with text and token arrays)
            if isinstance(llm_message, tuple) and len(llm_message) > 0:
                # Return only the text part, not the numpy arrays
                return str(llm_message[0])
            else:
                return str(llm_message)
            
        except Exception as e:
            print(f" Error in query_text_only: {e}")
            return f"Error: {str(e)}"
    
    def load_mm_safetybench_image(self, question_id, kind="kind3", category="01-Illegal_Activitiy"):
        """
        Load the actual MM-SafetyBench image for a given question.
        
        Args:
            question_id (str): Question ID (e.g., "0", "1", "2", ...)
            kind (str): Image-question kind ("kind1", "kind2", "kind3")
            category (str): Category name
            
        Returns:
            PIL.Image: The loaded image, or None if not found
        """
        # Map kinds to image directories
        kind_to_dir = {
            "kind1": "SD",        # Kind 1: SD images + "Rephrased Question(SD)" text
            "kind2": "SD_TYPO",   # Kind 2: SD_TYPO images + "Rephrased Question" text
            "kind3": "TYPO"       # Kind 3: TYPO images + "Rephrased Question" text
        }
        
        img_dir = kind_to_dir.get(kind, "TYPO")
        img_path = f"../data/MM-SafetyBench/imgs/{category}/{img_dir}/{question_id}.jpg"
        
        try:
            image = Image.open(img_path)
            print(f"Loaded image ({kind}): {img_path}")
            return image
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
            return None
    
    def query_with_image(self, question, question_id, kind="kind3", category="01-Illegal_Activitiy", max_new_tokens=500):
        """
        Query MiniGPT with the actual MM-SafetyBench image.
        
        Args:
            question (str): The question to ask
            question_id (str): Question ID to load the corresponding image
            kind (str): Image-question kind for proper pairing
            category (str): Category name for image selection
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Model response
        """
        if self.model is None:
            self.load_model()
        
        # Load the actual MM-SafetyBench image
        image = self.load_mm_safetybench_image(question_id, kind, category)
        
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
                num_beams=1,
                temperature=0.7,
                max_new_tokens=max_new_tokens,
                max_length=2000
            )
            
            # Extract text from MiniGPT response (it returns tuple with text and token arrays)
            if isinstance(llm_message, tuple) and len(llm_message) > 0:
                # Return only the text part, not the numpy arrays
                return str(llm_message[0])
            else:
                return str(llm_message)
            
        except Exception as e:
            print(f"Error in query_with_image: {e}")
            # Fallback to text-only if image processing fails
            print("Falling back to text-only approach...")
            return self.query_text_only(question, max_new_tokens)

def add_model_responses(input_file_path, output_file_path, model_name, responses_dict):
    """
    Add responses from a specific model to the JSON file.
    
    Args:
        input_file_path (str): Path to input JSON file
        output_file_path (str): Path to output JSON file  
        model_name (str): Name of the model
        responses_dict (dict): Dictionary with question_id as key and response text as value
    """
    
    # Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add responses for the specified model
    for question_id, response_text in responses_dict.items():
        if question_id in data:
            # Initialize ans field if it doesn't exist
            if "ans" not in data[question_id]:
                data[question_id]["ans"] = {}
            
            # Add the model response
            data[question_id]["ans"][model_name] = {
                "text": response_text
            }
        else:
            print(f"Warning: Question ID {question_id} not found in data")
    
    # Write the output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Added {model_name} responses and saved to {output_file_path}")

def create_answer_format_structure(input_file_path, output_file_path):
    """
    Create the answer format structure with empty response fields.
    
    Args:
        input_file_path (str): Path to input JSON file with questions
        output_file_path (str): Path to output JSON file with answer structure
    """
    
    # Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each question to add answer structure
    for question_id, question_data in data.items():
        # Initialize the ans field
        question_data["ans"] = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Write the output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Created answer format structure for {len(data)} questions")
    return data

def query_minigpt_for_category(category, use_images=True, kind="kind3", max_questions=None, model_key="minigpt4-vicuna-7b", minigpt_repo_path="../.cache/MiniGPT-4"):
    """
    Query MiniGPT for all questions in a specific category with correct image-question pairing.
    
    Args:
        category (str): Category name (e.g., "01-Illegal_Activitiy")
        use_images (bool): Whether to use images
        kind (str): Image-question kind ("kind1", "kind2", "kind3")
        max_questions (int): Maximum number of questions to process (None for all)
        model_key (str): Model key from AVAILABLE_MODELS
        minigpt_repo_path (str): Path to MiniGPT repository
        
    Returns:
        dict: Dictionary with question_id as key and response as value
    """
    # Load the questions for this category
    input_file = f"../data/MM-SafetyBench/processed_questions/{category}.json"
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found for category {category}")
        return {}
    
    # Initialize MiniGPT (create new instance for different models)
    cache_key = f"minigpt_engine_{model_key}"
    if not hasattr(query_minigpt_for_category, cache_key):
        setattr(query_minigpt_for_category, cache_key, MiniGPTQueryEngine(model_key, minigpt_repo_path))
    
    minigpt_engine = getattr(query_minigpt_for_category, cache_key)
    
    responses = {}
    question_ids = list(questions_data.keys())
    
    if max_questions:
        question_ids = question_ids[:max_questions]
    
    print(f"Processing {len(question_ids)} questions for {category} ({kind}) with {minigpt_engine.display_name}...")
    
    for i, question_id in enumerate(question_ids):
        print(f"  Processing question {i+1}/{len(question_ids)}: {question_id}")
        
        question_data = questions_data[question_id]
        
        # Choose which question to ask based on kind (correct pairing)
        if kind == "kind1":
            # Kind 1: SD images + "Rephrased Question(SD)" text
            question = question_data["Rephrased Question(SD)"]
        elif kind == "kind2" or kind == "kind3":
            # Kind 2 & 3: SD_TYPO/TYPO images + "Rephrased Question" text
            question = question_data["Rephrased Question"]
        else:
            # Fallback
            question = question_data["Rephrased Question"]
        
        try:
            if use_images:
                # Use the actual MM-SafetyBench image for this question
                response = minigpt_engine.query_with_image(
                    question, 
                    question_id,
                    kind,
                    category
                )
            else:
                # Use text-only query
                response = minigpt_engine.query_text_only(question)
            
            responses[question_id] = response
            print(f"  ✓ Response received for question {question_id}")
            
        except Exception as e:
            print(f"  ✗ Error processing question {question_id}: {str(e)}")
            responses[question_id] = f"Error: {str(e)}"
    
    return responses

def generate_complete_dataset_evaluation(use_images=True, max_questions_per_category=None, model_key="minigpt4-vicuna-7b", minigpt_repo_path="../.cache/MiniGPT-4"):
    """
    Complete MiniGPT safety evaluation across ALL MM-SafetyBench categories and kinds.
    
    Args:
        use_images (bool): Whether to use images
        max_questions_per_category (int): Maximum number of questions per category (None for all)
        model_key (str): Model key from AVAILABLE_MODELS
        minigpt_repo_path (str): Path to MiniGPT repository
    """
    model_info = get_model_info(model_key)
    model_display_name = model_info["display_name"]
    
    print("=" * 80)
    print(f"🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH {model_display_name}")
    print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
    print("=" * 80)
    print(f"🤖 Model: {model_display_name}")
    print(f"📂 MiniGPT Repo: {minigpt_repo_path}")
    print(f"📊 Use images: {use_images}")
    print(f"🔢 Max questions per category per kind: {max_questions_per_category or 'ALL'}")
    
    # Define all MM-SafetyBench categories
    categories = [
        "01-Illegal_Activitiy",
        "02-HateSpeech", 
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision"
    ]
    
    # Define all MM-SafetyBench kinds with proper image-question pairing
    kinds = [
        ("kind1", "SD", "Rephrased Question(SD)"),
        ("kind2", "SD_TYPO", "Rephrased Question"),
        ("kind3", "TYPO", "Rephrased Question")
    ]
    
    print(f"\n🗂️  Processing {len(categories)} categories × {len(kinds)} kinds = {len(categories) * len(kinds)} total combinations:")
    for i, cat in enumerate(categories, 1):
        print(f"   {i:2d}. {cat}")
    print(f"\n📝 Kinds to process:")
    for kind_id, img_dir, question_text in kinds:
        print(f"   • {kind_id}: {img_dir} images + '{question_text}' text")
    
    total_responses = 0
    all_results = {}
    
    # Process each category × each kind
    combination_count = 0
    total_combinations = len(categories) * len(kinds)
    
    for category_idx, category in enumerate(categories, 1):
        for kind_idx, (kind_id, img_dir, question_text) in enumerate(kinds, 1):
            combination_count += 1
            
            print(f"\n" + "="*80)
            print(f"📂 COMBINATION {combination_count}/{total_combinations}")
            print(f"📁 CATEGORY {category_idx}/{len(categories)}: {category}")  
            print(f"🖼️  KIND {kind_idx}/{len(kinds)}: {kind_id} ({img_dir} + '{question_text}')")
            print(f"🤖 MODEL: {model_display_name}")
            print("="*80)
            
            try:
                # Step 1: Create answer format structure for this category (only once per category)
                input_questions_file = f"../data/MM-SafetyBench/processed_questions/{category}.json"
                temp_structure_file = f"../data/MM-SafetyBench/answer_format/{category}_with_responses.json"
                
                # Create organized directory structure: model_name/image_type/category_responses.json
                output_dir = f"../data/MM-SafetyBench/question_and_answer/{model_display_name}/{img_dir}"
                os.makedirs(output_dir, exist_ok=True)
                final_output_file = f"{output_dir}/{category}_with_responses.json"
                
                if kind_idx == 1:  # Only create structure once per category
                    print(f"📝 Creating answer structure...")
                    create_answer_format_structure(input_questions_file, temp_structure_file)
                
                # Step 2: Query MiniGPT for this category + kind combination
                print(f"🤖 Querying {model_display_name} with {kind_id}...")
                category_responses = query_minigpt_for_category(
                    category=category,
                    use_images=use_images,
                    kind=kind_id,
                    max_questions=max_questions_per_category,
                    model_key=model_key,
                    minigpt_repo_path=minigpt_repo_path
                )
                
                # Step 3: Save results for this category + kind
                print(f"💾 Saving results...")
                add_model_responses(temp_structure_file, final_output_file, model_display_name, category_responses)
                
                # Track results
                result_key = f"{category}_{img_dir}"
                all_results[result_key] = {
                    'category': category,
                    'kind': kind_id,
                    'img_dir': img_dir,
                    'question_text': question_text,
                    'num_questions': len(category_responses),
                    'output_file': final_output_file,
                    'model': model_display_name
                }
                total_responses += len(category_responses)
                
                print(f"✅ {category} {kind_id} completed: {len(category_responses)} questions processed")
                
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
    print("\n" + "="*80)
    print("🎉 COMPLETE DATASET EVALUATION FINISHED!")
    print("🔄 ALL 3 KINDS × ALL 13 CATEGORIES PROCESSED")
    print("="*80)
    print(f"🤖 Model used: {model_display_name}")
    print(f"📊 Total questions processed: {total_responses}")
    print(f"🔢 Total combinations: {len(all_results)} (should be {len(categories)} × {len(kinds)} = {len(categories) * len(kinds)})")
    print(f"✅ Successful combinations: {len([r for r in all_results.values() if 'error' not in r])}")
    print(f"❌ Failed combinations: {len([r for r in all_results.values() if 'error' in r])}")
    
    print("\n📋 DETAILED RESULTS BY CATEGORY:")
    for category in categories:
        print(f"\n📁 {category}:")
        for kind_id, img_dir, _ in kinds:
            result_key = f"{category}_{img_dir}"
            if result_key in all_results:
                result = all_results[result_key]
                if 'error' in result:
                    print(f"   ❌ {img_dir}: ERROR - {result['error']}")
                else:
                    print(f"   ✅ {img_dir}: {result['num_questions']} questions → {result['output_file']}")
            else:
                print(f"   ⚠️  {img_dir}: NOT PROCESSED")
    
    print(f"\n🎯 All results saved in: ../data/MM-SafetyBench/question_and_answer/")
    print(f"📁 Directory structure: {{model_name}}/{{image_type}}/{{category}}_with_responses.json")
    print(f"    Examples: {model_display_name}/SD/01-Illegal_Activitiy_with_responses.json")
    print(f"              {model_display_name}/SD_TYPO/02-HateSpeech_with_responses.json")
    print(f"              {model_display_name}/TYPO/03-Malware_Generation_with_responses.json")
    print("="*80)
    
    return all_results

def interactive_model_selection():
    """Interactive model selection for users."""
    print_available_models()
    
    while True:
        choice = input("🔤 Enter model key (or 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            print("👋 Goodbye!")
            return None
        
        if choice in AVAILABLE_MODELS:
            model_info = AVAILABLE_MODELS[choice]
            print(f"\n✅ Selected: {model_info['display_name']}")
            print(f"📝 Description: {model_info['description']}")
            return choice
        else:
            print(f"❌ Invalid choice: '{choice}'")
            print(f"✅ Valid options: {', '.join(AVAILABLE_MODELS.keys())}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MM-SafetyBench evaluation with MiniGPT models",
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
  python add_minigpt_responses.py --model minigpt4-vicuna-7b --max-questions 5
  python add_minigpt_responses.py --model minigptv2-llama2 --text-only
  python add_minigpt_responses.py --interactive
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
        default="../.cache/MiniGPT-4",
        help="Path to the MiniGPT-4 repository (default: ../.cache/MiniGPT-4)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle special flags
    if args.list_models:
        print_available_models()
        exit(0)
    
    # Check if MiniGPT repository exists
    if not os.path.exists(args.minigpt_repo_path):
        print(f"❌ MiniGPT repository not found at: {args.minigpt_repo_path}")
        print("💡 Please clone the MiniGPT-4 repository:")
        print("   git clone https://github.com/Vision-CAIR/MiniGPT-4.git")
        print("   OR update --minigpt-repo-path to point to your MiniGPT-4 directory")
        exit(1)
    
    # Determine model to use
    if args.interactive:
        model_key = interactive_model_selection()
        if model_key is None:
            exit(0)
    else:
        model_key = args.model
    
    # Determine max questions (0 means all)
    max_questions = args.max_questions if args.max_questions > 0 else None
    
    # Run evaluation
    print("\n" + "="*80)
    print("🚀 STARTING MM-SAFETYBENCH EVALUATION WITH MINIGPT MODELS")
    print("="*80)
    print(f"🤖 Model: {AVAILABLE_MODELS[model_key]['display_name']}")
    print(f"📂 MiniGPT Repo: {args.minigpt_repo_path}")
    print(f"🖼️  Images: {'Disabled (text-only)' if args.text_only else 'Enabled'}")
    print(f"🔢 Questions per category: {max_questions or 'ALL'}")
    print("="*80)
    
    # Complete MiniGPT safety evaluation across ALL MM-SafetyBench categories and kinds
    generate_complete_dataset_evaluation(
        use_images=not args.text_only,
        max_questions_per_category=max_questions,
        model_key=model_key,
        minigpt_repo_path=args.minigpt_repo_path
    )
    
    print("\n" + "="*80)
    print("🔧 QUICK REFERENCE")
    print("="*80)
    print("📋 Available models:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  --model {key:20} → {info['display_name']}")
    print("\n🎛️  Command-line options:")
    print("  --model <model_key>        Choose MiniGPT model")
    print("  --max-questions <N>        Process N questions per category (0 = all)")
    print("  --text-only                Skip images, use text-only queries")
    print("  --interactive              Interactive model selection")
    print("  --list-models              Show available models")
    print("  --minigpt-repo-path <path> Path to MiniGPT-4 repository")
    print("\n📝 Examples:")
    print("  python add_minigpt_responses.py --model minigpt4-vicuna-13b --max-questions 10")
    print("  python add_minigpt_responses.py --interactive")
    print("="*80)
