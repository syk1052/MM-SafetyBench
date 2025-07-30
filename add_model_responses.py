import json
import os
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image

class LlavaQueryEngine:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA model for querying.
        
        Args:
            model_name (str): HuggingFace model name for LLaVA
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the LLaVA model and processor."""
        print(f"Loading LLaVA model: {self.model_name}")
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

def query_llava_for_category(category, use_images=True, kind="kind3", max_questions=None):
    """
    Query LLaVA-1.5(7B) for all questions in a specific category with correct image-question pairing.
    
    Args:
        category (str): Category name (e.g., "01-Illegal_Activitiy")
        use_images (bool): Whether to use images
        kind (str): Image-question kind ("kind1", "kind2", "kind3")
        max_questions (int): Maximum number of questions to process (None for all)
        
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
    
    # Initialize LLaVA (reuse existing instance if available)
    if not hasattr(query_llava_for_category, 'llava_engine'):
        query_llava_for_category.llava_engine = LlavaQueryEngine()
    
    llava_engine = query_llava_for_category.llava_engine
    
    responses = {}
    question_ids = list(questions_data.keys())
    
    if max_questions:
        question_ids = question_ids[:max_questions]
    
    print(f"Processing {len(question_ids)} questions for {category} ({kind})...")
    
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

def generate_complete_dataset_evaluation(use_images=True, max_questions_per_category=None):
    """
    Complete LLaVA-1.5(7B) safety evaluation across ALL MM-SafetyBench categories and kinds.
    
    Args:
        use_images (bool): Whether to use images
        max_questions_per_category (int): Maximum number of questions per category (None for all)
    """
    print("=" * 80)
    print("🚀 COMPLETE MM-SAFETYBENCH EVALUATION WITH LLaVA-1.5(7B)")
    print("🔄 PROCESSING ALL 3 KINDS (SD, SD_TYPO, TYPO) × ALL 13 CATEGORIES")
    print("=" * 80)
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
            print("="*80)
            
            try:
                # Step 1: Create answer format structure for this category (only once per category)
                input_questions_file = f"../data/MM-SafetyBench/processed_questions/{category}.json"
                temp_structure_file = f"../data/MM-SafetyBench/answer_format/{category}_with_responses.json"
                final_output_file = f"../data/MM-SafetyBench/question_and_answer/{category}_{img_dir}_with_llava_responses.json"
                
                if kind_idx == 1:  # Only create structure once per category
                    print(f"📝 Creating answer structure...")
                    create_answer_format_structure(input_questions_file, temp_structure_file)
                
                # Step 2: Query LLaVA for this category + kind combination
                print(f"🤖 Querying LLaVA-1.5(7B) with {kind_id}...")
                category_responses = query_llava_for_category(
                    category=category,
                    use_images=use_images,
                    kind=kind_id,
                    max_questions=max_questions_per_category
                )
                
                # Step 3: Save results for this category + kind
                print(f"💾 Saving results...")
                add_model_responses(temp_structure_file, final_output_file, "LLaVA-1.5(7B)", category_responses)
                
                # Track results
                result_key = f"{category}_{img_dir}"
                all_results[result_key] = {
                    'category': category,
                    'kind': kind_id,
                    'img_dir': img_dir,
                    'question_text': question_text,
                    'num_questions': len(category_responses),
                    'output_file': final_output_file
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
                    'error': str(e)
                }
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 COMPLETE DATASET EVALUATION FINISHED!")
    print("🔄 ALL 3 KINDS × ALL 13 CATEGORIES PROCESSED")
    print("="*80)
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
    
    print("\n🎯 All results saved in: ../data/MM-SafetyBench/question_and_answer/")
    print("📁 File naming: {category}_{img_dir}_with_llava_responses.json")
    print("    Examples: 01-Illegal_Activitiy_SD_with_llava_responses.json")
    print("              01-Illegal_Activitiy_SD_TYPO_with_llava_responses.json")
    print("              01-Illegal_Activitiy_TYPO_with_llava_responses.json")
    print("="*80)
    
    return all_results



if __name__ == "__main__":
    # Complete LLaVA-1.5(7B) safety evaluation across ALL MM-SafetyBench categories and kinds
    generate_complete_dataset_evaluation(
        use_images=True,                    # Use actual MM-SafetyBench images
        max_questions_per_category=5        # Start with 5 questions per category per kind (change to None for full dataset)
    )
    
    print("\n" + "="*80)
    print("🔧 CONFIGURATION OPTIONS")
    print("="*80)
    print("• To process ALL questions in dataset:")
    print("    change max_questions_per_category=5 to max_questions_per_category=None")
    print("• Image-Question pairings (automatically processed):")
    print("    - Kind 1: SD images + 'Rephrased Question(SD)' text")
    print("    - Kind 2: SD_TYPO images + 'Rephrased Question' text")
    print("    - Kind 3: TYPO images + 'Rephrased Question' text")
    print("• Image usage:")
    print("    - use_images=True: Use actual MM-SafetyBench images (recommended)")
    print("    - use_images=False: Text-only queries")
    print("• Processing scope:")
    print("    - ALL 13 MM-SafetyBench safety categories")
    print("    - ALL 3 image-question kinds per category (SD, SD_TYPO, TYPO)")
    print("    - Results saved per category-kind combination with descriptive names")
    print("="*80) 