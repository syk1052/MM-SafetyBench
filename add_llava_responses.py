import json
import os
import torch
import argparse
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

# Available LLaVA models
AVAILABLE_MODELS = {
    "llava-1.5-7b": {
        "hf_name": "llava-hf/llava-1.5-7b-hf",
        "display_name": "LLaVA-1.5(7B)",
        "description": "LLaVA 1.5 with 7B parameters (faster, less memory)"
    },
    "llava-1.5-13b": {
        "hf_name": "llava-hf/llava-1.5-13b-hf", 
        "display_name": "LLaVA-1.5(13B)",
        "description": "LLaVA 1.5 with 13B parameters (slower, more memory, potentially better performance)"
    }
}

def print_available_models():
    """Print available LLaVA models for selection."""
    print("\n🤖 Available LLaVA Models:")
    print("="*60)
    for key, info in AVAILABLE_MODELS.items():
        print(f"  {key:15} → {info['display_name']}")
        print(f"  {' '*15}   {info['description']}")
        print()

def get_model_info(model_key):
    """Get model information from the model key."""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_key]

class LlavaQueryEngine:
    def __init__(self, model_key="llava-1.5-7b"):
        """
        Initialize LLaVA model for querying.
        
        Args:
            model_key (str): Model key from AVAILABLE_MODELS
        """
        self.model_info = get_model_info(model_key)
        self.model_name = self.model_info["hf_name"]
        self.display_name = self.model_info["display_name"]
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Selected model: {self.display_name}")
        
    def load_model(self):
        """Load the LLaVA model and processor."""
        print(f"Loading LLaVA model: {self.display_name}")
        print(f"HuggingFace model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        print("Model loaded successfully!")
    

    
    def query_text_only(self, question, max_new_tokens=500):
        """
        Query LLaVA with text only (no image).
        
        Args:
            question (str): The question to ask
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: Model response
        """
        if self.model is None:
            self.load_model()
        
        prompt = f"USER: {question}\nASSISTANT:"
        
        try:
            # Use text-only approach that we know works
            inputs = self.processor(text=prompt, images=None, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode and extract the response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
        except Exception as e:
            print(f"Error in query_text_only: {e}")
            return f"Error: {str(e)}"
    
    def load_mm_safetybench_image(self, question_id, kind="kind3", category="01-Illegal_Activitiy"):
        """
        Load the actual MM-SafetyBench image for a given question.
        
        Args:
            question_id (str): Question ID (e.g., "0", "1", "2", ...)
            kind (str): Image-question kind ("kind1", "kind2", "kind3")
                - kind1: SD images + "Rephrased Question(SD)" text
                - kind2: SD_TYPO images + "Rephrased Question" text  
                - kind3: TYPO images + "Rephrased Question" text
            category (str): Category name (e.g., "01-Illegal_Activitiy", "02-HateSpeech", etc.)
            
        Returns:
            PIL.Image: The loaded image, or None if not found
        """
        # Map kinds to image directories (corrected mapping)
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
        Query LLaVA with the actual MM-SafetyBench image.
        
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
        
        # Use the proper LLaVA format with image token
        prompt = f"<image>\nUSER: {question}\nASSISTANT:"
        
        try:
            # Process the image and text separately then combine (image as list)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            
            # Move inputs to device, handling each tensor individually
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode and extract the response
            response = self.processor.decode(output[0], skip_special_tokens=True)
            # Extract only the assistant's response
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            return response
            
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

def load_model_responses_from_file(responses_file_path):
    """
    Load model responses from a text file or JSON file.
    
    Args:
        responses_file_path (str): Path to file containing model responses
        
    Returns:
        dict: Dictionary with question_id as key and response text as value
    """
    responses = {}
    
    if responses_file_path.endswith('.json'):
        # Load from JSON file
        with open(responses_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for question_id, response in data.items():
                responses[question_id] = response
    else:
        # Load from text file (assuming format: question_id|response)
        with open(responses_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    question_id, response = line.strip().split('|', 1)
                    responses[question_id] = response
    
    return responses

def query_llava_for_category(category, use_images=True, kind="kind3", max_questions=None, model_key="llava-1.5-7b"):
    """
    Query LLaVA for all questions in a specific category with correct image-question pairing.
    
    Args:
        category (str): Category name (e.g., "01-Illegal_Activitiy")
        use_images (bool): Whether to use images
        kind (str): Image-question kind ("kind1", "kind2", "kind3")
        max_questions (int): Maximum number of questions to process (None for all)
        model_key (str): Model key from AVAILABLE_MODELS
        
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
    
    # Initialize LLaVA (create new instance for different models)
    cache_key = f"llava_engine_{model_key}"
    if not hasattr(query_llava_for_category, cache_key):
        setattr(query_llava_for_category, cache_key, LlavaQueryEngine(model_key))
    
    llava_engine = getattr(query_llava_for_category, cache_key)
    
    responses = {}
    question_ids = list(questions_data.keys())
    
    if max_questions:
        question_ids = question_ids[:max_questions]
    
    print(f"Processing {len(question_ids)} questions for {category} ({kind}) with {llava_engine.display_name}...")
    
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
                response = llava_engine.query_with_image(
                    question, 
                    question_id,
                    kind,
                    category
                )
            else:
                # Use text-only query
                response = llava_engine.query_text_only(question)
            
            responses[question_id] = response
            print(f"  ✓ Response received for question {question_id}")
            
        except Exception as e:
            print(f"  ✗ Error processing question {question_id}: {str(e)}")
            responses[question_id] = f"Error: {str(e)}"
    
    return responses

def generate_complete_dataset_evaluation(use_images=True, max_questions_per_category=None, model_key="llava-1.5-7b"):
    """
    Complete LLaVA safety evaluation across ALL MM-SafetyBench categories and kinds.
    
    Args:
        use_images (bool): Whether to use images
        max_questions_per_category (int): Maximum number of questions per category (None for all)
        model_key (str): Model key from AVAILABLE_MODELS
    """
    model_info = get_model_info(model_key)
    model_display_name = model_info["display_name"]
    
    print("=" * 80)
    print(f"🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH {model_display_name}")
    print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
    print("=" * 80)
    print(f"🤖 Model: {model_display_name}")
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
                
                # Step 2: Query LLaVA for this category + kind combination
                print(f"🤖 Querying {model_display_name} with {kind_id}...")
                category_responses = query_llava_for_category(
                    category=category,
                    use_images=use_images,
                    kind=kind_id,
                    max_questions=max_questions_per_category,
                    model_key=model_key
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
        description="MM-SafetyBench evaluation with LLaVA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Models:
  llava-1.5-7b   → LLaVA-1.5(7B)  - Faster, less memory
  llava-1.5-13b  → LLaVA-1.5(13B) - Slower, more memory, potentially better

Examples:
  python add_llava_responses.py --model llava-1.5-7b --max-questions 5
  python add_llava_responses.py --model llava-1.5-13b --text-only
  python add_llava_responses.py --interactive
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
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle special flags
    if args.list_models:
        print_available_models()
        exit(0)
    
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
    print("🚀 STARTING MM-SAFETYBENCH EVALUATION WITH LLAVA MODELS")
    print("="*80)
    print(f"🤖 Model: {AVAILABLE_MODELS[model_key]['display_name']}")
    print(f"🖼️  Images: {'Disabled (text-only)' if args.text_only else 'Enabled'}")
    print(f"🔢 Questions per category: {max_questions or 'ALL'}")
    print("="*80)
    
    # Complete LLaVA safety evaluation across ALL MM-SafetyBench categories and kinds
    generate_complete_dataset_evaluation(
        use_images=not args.text_only,
        max_questions_per_category=max_questions,
        model_key=model_key
    )
    
    print("\n" + "="*80)
    print("🔧 QUICK REFERENCE")
    print("="*80)
    print("📋 Available models:")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  --model {key:15} → {info['display_name']}")
    print("\n🎛️  Command-line options:")
    print("  --model <model_key>     Choose LLaVA model")
    print("  --max-questions <N>     Process N questions per category (0 = all)")
    print("  --text-only             Skip images, use text-only queries")
    print("  --interactive           Interactive model selection")
    print("  --list-models           Show available models")
    print("\n📝 Examples:")
    print("  python add_llava_responses.py --model llava-1.5-13b --max-questions 10")
    print("  python add_llava_responses.py --interactive")
    print("="*80) 