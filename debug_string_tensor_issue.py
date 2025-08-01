#!/usr/bin/env python3
"""
Debug script to investigate the 'str' object has no attribute 'shape' error in CogVLM
"""

import torch
from PIL import Image
import traceback
from transformers import LlamaTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import sys

def debug_string_tensor_issue():
    """Debug the specific 'str' object has no attribute 'shape' error."""
    print("=== DEBUGGING STRING/TENSOR ISSUE IN COGVLM ===")
    
    try:
        # Load model exactly as the optimized script does
        print("1. Loading CogVLM model...")
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on device: {device}")
        
        # Load test image
        print("\n2. Loading test image...")
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        print(f"Image: {image.size} - {image.mode}")
        
        # Test query
        print("\n3. Building conversation inputs...")
        query = '<EOI>Question: What do you see in this image? Answer:'
        print(f"Query: {query}")
        
        # Build inputs using CogVLM's method
        inputs = model.build_conversation_input_ids(
            tokenizer, 
            query=query, 
            history=[], 
            images=[image]
        )
        
        print(f"\n4. Raw inputs inspection:")
        print(f"Keys: {list(inputs.keys())}")
        
        for key, value in inputs.items():
            print(f"\n--- {key} ---")
            print(f"Type: {type(value)}")
            
            if isinstance(value, list):
                print(f"List length: {len(value)}")
                for i, item in enumerate(value):
                    print(f"  [{i}] Type: {type(item)}")
                    if torch.is_tensor(item):
                        print(f"      Shape: {item.shape}")
                        print(f"      Dtype: {item.dtype}")
                        print(f"      Device: {item.device}")
                    elif isinstance(item, str):
                        print(f"      String content: {repr(item[:100])}...")
                    else:
                        print(f"      Value: {item}")
                        
            elif torch.is_tensor(value):
                print(f"Shape: {value.shape}")
                print(f"Dtype: {value.dtype}")
                print(f"Device: {value.device}")
                
            elif isinstance(value, str):
                print(f"String content: {repr(value[:100])}...")
                
            else:
                print(f"Value: {value}")
        
        # Test the input cleaning process step by step
        print(f"\n5. Testing input cleaning process...")
        cleaned_inputs = clean_inputs_debug(inputs, device)
        
        print(f"\n6. Cleaned inputs inspection:")
        for key, value in cleaned_inputs.items():
            print(f"\n--- {key} (cleaned) ---")
            print(f"Type: {type(value)}")
            
            if isinstance(value, list):
                print(f"List length: {len(value)}")
                for i, item in enumerate(value):
                    print(f"  [{i}] Type: {type(item)}")
                    if torch.is_tensor(item):
                        print(f"      Shape: {item.shape}")
                        print(f"      Dtype: {item.dtype}")
                        print(f"      Device: {item.device}")
                    else:
                        print(f"      Value: {item}")
                        
            elif torch.is_tensor(value):
                print(f"Shape: {value.shape}")
                print(f"Dtype: {value.dtype}")
                print(f"Device: {value.device}")
            else:
                print(f"Value: {value}")
        
        # Test model.generate step by step
        print(f"\n7. Testing model.generate step by step...")
        
        # First, let's test what happens when we call generate
        try:
            print("7a. Testing with cleaned inputs...")
            with torch.no_grad():
                outputs = model.generate(
                    **cleaned_inputs,
                    max_new_tokens=10,  # Very small for testing
                    do_sample=False,
                    use_cache=False
                )
            print(f"✅ Success with cleaned inputs! Output shape: {outputs.shape}")
            
        except Exception as e:
            print(f"❌ Failed with cleaned inputs: {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Let's debug step by step what happens in generate
            print(f"\n7b. Debugging generate internals...")
            debug_generate_internals(model, cleaned_inputs, tokenizer, device)
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

def clean_inputs_debug(inputs, device):
    """Debug version of input cleaning to see what's happening."""
    print(f"\n--- CLEANING INPUTS DEBUG ---")
    cleaned_inputs = {}
    
    for key, value in inputs.items():
        print(f"\nProcessing key: {key}")
        print(f"Original type: {type(value)}")
        
        # Skip known problematic string fields
        if isinstance(value, str):
            print(f"  → Skipping string field: {repr(value[:50])}...")
            continue
            
        # Handle list/tuple fields (like images)
        elif isinstance(value, (list, tuple)):
            print(f"  → Processing list/tuple with {len(value)} items")
            cleaned_list = []
            for i, item in enumerate(value):
                print(f"    Item [{i}]: {type(item)}")
                if torch.is_tensor(item):
                    # Ensure proper dimensions and device
                    print(f"      Original tensor: {item.shape}, {item.dtype}, {item.device}")
                    if item.dim() == 3:
                        item = item.unsqueeze(0)  # Add batch dimension
                        print(f"      Added batch dim: {item.shape}")
                    item = item.to(device)
                    if device == "cuda" and item.dtype == torch.float32:
                        item = item.to(torch.bfloat16)
                        print(f"      Converted to bfloat16: {item.dtype}")
                    cleaned_list.append(item)
                    print(f"      Final tensor: {item.shape}, {item.dtype}, {item.device}")
                elif isinstance(item, str):
                    print(f"      Skipping string item: {repr(item[:50])}...")
                    continue
                else:
                    print(f"      Adding other item: {type(item)}")
                    cleaned_list.append(item)
            
            if cleaned_list:  # Only add if we have valid items
                cleaned_inputs[key] = cleaned_list
                print(f"  → Added {len(cleaned_list)} items to cleaned list")
            else:
                print(f"  → No valid items, skipping key")
                
        # Handle tensor fields
        elif torch.is_tensor(value):
            print(f"  → Processing tensor: {value.shape}, {value.dtype}, {value.device}")
            if value.dim() == 1:
                value = value.unsqueeze(0)  # Add batch dimension
                print(f"    Added batch dim: {value.shape}")
            value = value.to(device)
            cleaned_inputs[key] = value
            print(f"    Final tensor: {value.shape}, {value.dtype}, {value.device}")
            
        # Skip other non-tensor types that might cause issues
        elif hasattr(value, 'shape'):
            print(f"  → Adding object with shape: {type(value)}")
            cleaned_inputs[key] = value
        else:
            print(f"  → Skipping non-tensor field: {type(value)}")
            
    return cleaned_inputs

def debug_generate_internals(model, inputs, tokenizer, device):
    """Debug the internal steps of model.generate to find where strings appear."""
    print(f"\n--- DEBUGGING GENERATE INTERNALS ---")
    
    try:
        # Let's check what prepare_inputs_for_generation returns
        print(f"1. Testing prepare_inputs_for_generation...")
        
        if hasattr(model, 'prepare_inputs_for_generation'):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask')
            token_type_ids = inputs.get('token_type_ids')
            
            print(f"   input_ids: {input_ids.shape if torch.is_tensor(input_ids) else type(input_ids)}")
            print(f"   attention_mask: {attention_mask.shape if torch.is_tensor(attention_mask) else type(attention_mask)}")
            print(f"   token_type_ids: {token_type_ids.shape if torch.is_tensor(token_type_ids) else type(token_type_ids)}")
            
            try:
                prepared = model.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    **{k: v for k, v in inputs.items() if k not in ['input_ids', 'attention_mask', 'token_type_ids']}
                )
                
                print(f"   ✅ prepare_inputs_for_generation succeeded")
                print(f"   Prepared keys: {list(prepared.keys())}")
                
                for key, value in prepared.items():
                    if isinstance(value, str):
                        print(f"   ⚠️  STRING FOUND in prepared inputs: {key} = {repr(value[:50])}...")
                    elif torch.is_tensor(value):
                        print(f"   {key}: {value.shape}, {value.dtype}")
                    elif isinstance(value, list):
                        print(f"   {key}: list with {len(value)} items")
                        for i, item in enumerate(value):
                            if isinstance(item, str):
                                print(f"     ⚠️  STRING FOUND in list: [{i}] = {repr(item[:50])}...")
                            elif torch.is_tensor(item):
                                print(f"     [{i}]: {item.shape}, {item.dtype}")
                
            except Exception as e:
                print(f"   ❌ prepare_inputs_for_generation failed: {e}")
                traceback.print_exc()
        
        # Let's try a direct forward pass
        print(f"\n2. Testing direct forward pass...")
        try:
            with torch.no_grad():
                forward_output = model(**inputs, return_dict=True)
            print(f"   ✅ Direct forward pass succeeded")
            
        except Exception as e:
            print(f"   ❌ Direct forward pass failed: {e}")
            if "'str' object has no attribute 'shape'" in str(e):
                print(f"   🎯 Found the string/tensor issue!")
                
                # Let's inspect the model's forward method step by step
                print(f"\n3. Investigating model forward internals...")
                debug_model_forward_internals(model, inputs)
    
    except Exception as e:
        print(f"❌ Error in generate internals debug: {e}")
        traceback.print_exc()

def debug_model_forward_internals(model, inputs):
    """Debug the model's forward method to find where strings are being processed as tensors."""
    print(f"\n--- DEBUGGING MODEL FORWARD INTERNALS ---")
    
    try:
        # The error likely happens in the model's forward method
        # Let's check if it's in the vision processing or text processing
        
        print(f"1. Testing vision model separately...")
        if 'images' in inputs and inputs['images']:
            images = inputs['images']
            print(f"   Images type: {type(images)}")
            
            if isinstance(images, list):
                for i, img in enumerate(images):
                    print(f"   Image [{i}]: {type(img)}")
                    if torch.is_tensor(img):
                        print(f"     Shape: {img.shape}, dtype: {img.dtype}")
                    elif isinstance(img, str):
                        print(f"     ⚠️  STRING IMAGE: {repr(img[:100])}...")
                
                # Test vision model directly
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'vision'):
                        vision_model = model.model.vision
                        print(f"   Testing vision model with images...")
                        
                        # Prepare images properly for vision model
                        vision_inputs = []
                        for img in images:
                            if torch.is_tensor(img):
                                if img.dim() == 3:
                                    img = img.unsqueeze(0)
                                vision_inputs.append(img)
                            else:
                                print(f"     ⚠️  Non-tensor image found: {type(img)}")
                        
                        if vision_inputs:
                            vision_output = vision_model(vision_inputs[0])
                            print(f"   ✅ Vision model succeeded: {vision_output.shape}")
                        
                except Exception as ve:
                    print(f"   ❌ Vision model failed: {ve}")
                    traceback.print_exc()
        
        print(f"\n2. Testing LLM components...")
        
        # Check the text inputs
        text_inputs = {k: v for k, v in inputs.items() if k != 'images'}
        print(f"   Text input keys: {list(text_inputs.keys())}")
        
        for key, value in text_inputs.items():
            if isinstance(value, str):
                print(f"   ⚠️  STRING in text inputs: {key} = {repr(value[:50])}...")
            elif torch.is_tensor(value):
                print(f"   {key}: {value.shape}, {value.dtype}")
        
        # Try to identify the specific method that's causing the issue
        print(f"\n3. Testing encode_images method...")
        if 'images' in inputs and hasattr(model.model, 'encode_images'):
            try:
                images = inputs['images']
                print(f"   Testing encode_images with: {type(images)}")
                
                # The issue might be in how images are passed to encode_images
                if isinstance(images, list):
                    print(f"   Processing {len(images)} images...")
                    for i, img in enumerate(images):
                        print(f"     Image [{i}]: {type(img)}")
                        if isinstance(img, str):
                            print(f"       ⚠️  STRING IMAGE DETECTED: {repr(img[:100])}...")
                            print(f"       This is likely the cause of the error!")
                            
                            # Let's see where this string comes from
                            print(f"       Investigating string image source...")
                            investigate_string_image_source(model, img)
                            
                        elif torch.is_tensor(img):
                            print(f"       Tensor image: {img.shape}, {img.dtype}")
                
                # Try encode_images
                encoded = model.model.encode_images(images)
                print(f"   ✅ encode_images succeeded: {encoded.shape}")
                
            except Exception as ee:
                print(f"   ❌ encode_images failed: {ee}")
                if "'str' object has no attribute 'shape'" in str(ee):
                    print(f"   🎯 FOUND THE ISSUE! The error is in encode_images method")
                    print(f"   The images list contains strings instead of tensors")
                traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error in model forward internals debug: {e}")
        traceback.print_exc()

def investigate_string_image_source(model, string_image):
    """Investigate where the string image comes from."""
    print(f"\n--- INVESTIGATING STRING IMAGE SOURCE ---")
    
    print(f"String content: {repr(string_image[:200])}...")
    print(f"String length: {len(string_image)}")
    
    # Check if it's a file path
    if string_image.startswith('/') or string_image.startswith('./') or string_image.startswith('../'):
        print(f"Looks like a file path!")
        
    # Check if it's base64 encoded
    elif string_image.startswith('data:image/'):
        print(f"Looks like base64 encoded image!")
        
    # Check if it's some kind of token or special string
    elif '<' in string_image and '>' in string_image:
        print(f"Looks like it contains special tokens!")
        
    # This suggests the issue is in build_conversation_input_ids
    print(f"\nThe issue is likely in build_conversation_input_ids method")
    print(f"It's returning strings in the 'images' list instead of tensors")

def test_build_conversation_input_ids_debug():
    """Test the build_conversation_input_ids method in detail."""
    print("\n=== TESTING BUILD_CONVERSATION_INPUT_IDS DEBUG ===")
    
    try:
        # Load model
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm-chat-hf",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        # Load image
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Test different ways of calling build_conversation_input_ids
        print(f"1. Testing with PIL Image...")
        query = '<EOI>Question: What do you see? Answer:'
        
        try:
            inputs1 = model.build_conversation_input_ids(
                tokenizer, 
                query=query, 
                history=[], 
                images=[image]
            )
            
            print(f"   ✅ build_conversation_input_ids succeeded")
            print(f"   Keys: {list(inputs1.keys())}")
            
            if 'images' in inputs1:
                images_result = inputs1['images']
                print(f"   Images result type: {type(images_result)}")
                
                if isinstance(images_result, list):
                    for i, img in enumerate(images_result):
                        print(f"     Image [{i}]: {type(img)}")
                        if isinstance(img, str):
                            print(f"       ⚠️  STRING IMAGE: {repr(img[:100])}...")
                        elif torch.is_tensor(img):
                            print(f"       Tensor: {img.shape}, {img.dtype}")
                        else:
                            print(f"       Other: {img}")
            
        except Exception as e:
            print(f"   ❌ build_conversation_input_ids failed: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error in build_conversation_input_ids debug: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_string_tensor_issue()
    test_build_conversation_input_ids_debug() 