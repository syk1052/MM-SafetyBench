#!/usr/bin/env python3
"""
Deep dive debug script to investigate and fix the CogVLM string/tensor issue
"""

import torch
from PIL import Image
import traceback
from transformers import LlamaTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
import sys
import inspect
from functools import wraps

def intercept_method(obj, method_name, label=""):
    """Intercept and debug a method call."""
    if not hasattr(obj, method_name):
        return
        
    original_method = getattr(obj, method_name)
    
    @wraps(original_method)
    def intercepted_method(*args, **kwargs):
        print(f"\n🔍 INTERCEPTING {label}.{method_name}")
        print(f"   Args: {len(args)} args")
        print(f"   Kwargs: {list(kwargs.keys())}")
        
        # Inspect arguments for strings
        for i, arg in enumerate(args):
            if isinstance(arg, str):
                print(f"   ⚠️  STRING ARG[{i}]: {repr(arg[:100])}...")
            elif isinstance(arg, (list, tuple)):
                for j, item in enumerate(arg):
                    if isinstance(item, str):
                        print(f"   ⚠️  STRING in ARG[{i}][{j}]: {repr(item[:100])}...")
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                print(f"   ⚠️  STRING KWARG[{key}]: {repr(value[:100])}...")
            elif isinstance(value, (list, tuple)):
                for j, item in enumerate(value):
                    if isinstance(item, str):
                        print(f"   ⚠️  STRING in KWARG[{key}][{j}]: {repr(item[:100])}...")
        
        try:
            result = original_method(*args, **kwargs)
            print(f"   ✅ {method_name} succeeded")
            
            # Check result for strings
            if isinstance(result, str):
                print(f"   ⚠️  STRING RESULT: {repr(result[:100])}...")
            elif isinstance(result, (list, tuple)):
                for j, item in enumerate(result):
                    if isinstance(item, str):
                        print(f"   ⚠️  STRING in RESULT[{j}]: {repr(item[:100])}...")
            
            return result
            
        except Exception as e:
            print(f"   ❌ {method_name} failed: {e}")
            if "'str' object has no attribute 'shape'" in str(e):
                print(f"   🎯 FOUND THE STRING/TENSOR ISSUE!")
                # Let's examine the call stack
                print(f"   📍 Call stack analysis:")
                for frame_info in traceback.extract_tb(e.__traceback__):
                    print(f"      {frame_info.filename}:{frame_info.lineno} in {frame_info.name}")
                    if frame_info.line:
                        print(f"        {frame_info.line}")
            raise e
    
    setattr(obj, method_name, intercepted_method)

def deep_debug_cogvlm():
    """Deep dive debug of CogVLM generation pipeline."""
    print("=== DEEP DIVE COGVLM DEBUG ===")
    
    try:
        # Load model
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
        
        # Set up comprehensive interception
        print("\n2. Setting up method interception...")
        
        # Intercept key methods in the generation pipeline
        intercept_method(model, 'generate', 'model')
        intercept_method(model, 'prepare_inputs_for_generation', 'model')
        intercept_method(model, '_extract_past_from_model_output', 'model')
        intercept_method(model, 'forward', 'model')
        
        if hasattr(model, 'model'):
            intercept_method(model.model, 'forward', 'model.model')
            intercept_method(model.model, 'encode_images', 'model.model')
            
            if hasattr(model.model, 'vision'):
                intercept_method(model.model.vision, 'forward', 'vision')
                
                if hasattr(model.model.vision, 'patch_embedding'):
                    intercept_method(model.model.vision.patch_embedding, 'forward', 'patch_embedding')
        
        # Load test image and prepare inputs
        print("\n3. Preparing test inputs...")
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        
        query = '<EOI>Question: What do you see in this image? Answer:'
        
        inputs = model.build_conversation_input_ids(
            tokenizer, 
            query=query, 
            history=[], 
            images=[image]
        )
        
        # Clean inputs
        cleaned_inputs = clean_inputs_with_debug(inputs, device)
        
        print("\n4. Testing generation with full interception...")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **cleaned_inputs,
                    max_new_tokens=5,  # Very small for focused debugging
                    do_sample=False,
                    use_cache=False
                )
            print(f"✅ Generation succeeded! Outputs: {outputs.shape}")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            
            if "'str' object has no attribute 'shape'" in str(e):
                print(f"\n🎯 ANALYZING STRING/TENSOR ERROR...")
                analyze_string_tensor_error(e, model, cleaned_inputs, tokenizer)
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

def clean_inputs_with_debug(inputs, device):
    """Clean inputs with detailed debugging."""
    print(f"\n--- CLEANING INPUTS WITH DEBUG ---")
    cleaned_inputs = {}
    
    for key, value in inputs.items():
        print(f"\nProcessing {key}: {type(value)}")
        
        if isinstance(value, str):
            print(f"  → Skipping string: {repr(value[:50])}...")
            continue
            
        elif isinstance(value, (list, tuple)):
            cleaned_list = []
            for i, item in enumerate(value):
                print(f"  Item[{i}]: {type(item)}")
                if torch.is_tensor(item):
                    print(f"    Original: {item.shape}, {item.dtype}, {item.device}")
                    if item.dim() == 3:
                        item = item.unsqueeze(0)
                        print(f"    Added batch: {item.shape}")
                    item = item.to(device)
                    if item.dtype == torch.float32:
                        item = item.to(torch.bfloat16)
                        print(f"    Converted dtype: {item.dtype}")
                    cleaned_list.append(item)
                    print(f"    Final: {item.shape}, {item.dtype}, {item.device}")
                elif isinstance(item, str):
                    print(f"    ⚠️  String item: {repr(item[:50])}...")
                    continue
                else:
                    cleaned_list.append(item)
                    print(f"    Other: {type(item)}")
            
            if cleaned_list:
                cleaned_inputs[key] = cleaned_list
                print(f"  → Added list with {len(cleaned_list)} items")
                
        elif torch.is_tensor(value):
            print(f"  Tensor: {value.shape}, {value.dtype}, {value.device}")
            if value.dim() == 1:
                value = value.unsqueeze(0)
                print(f"  Added batch: {value.shape}")
            value = value.to(device)
            cleaned_inputs[key] = value
            
        else:
            print(f"  Other: {type(value)}")
    
    return cleaned_inputs

def analyze_string_tensor_error(error, model, inputs, tokenizer):
    """Analyze the specific string/tensor error in detail."""
    print(f"\n=== STRING/TENSOR ERROR ANALYSIS ===")
    
    # Get the traceback
    tb = error.__traceback__
    frames = traceback.extract_tb(tb)
    
    print(f"Error: {error}")
    print(f"Traceback frames: {len(frames)}")
    
    for i, frame in enumerate(frames):
        print(f"\nFrame {i}: {frame.filename}:{frame.lineno}")
        print(f"  Function: {frame.name}")
        print(f"  Line: {frame.line}")
        
        # Look for specific patterns that might indicate the issue
        if "'str' object has no attribute 'shape'" in str(error):
            if "past_key_values" in str(frame.line) or "kv" in str(frame.line):
                print(f"  🎯 Likely issue: past_key_values contains strings!")
                
            elif "attention" in str(frame.line):
                print(f"  🎯 Likely issue: attention mechanism receiving strings!")
                
            elif "hidden_states" in str(frame.line):
                print(f"  🎯 Likely issue: hidden_states contains strings!")
    
    # Let's try to isolate the specific problematic component
    print(f"\n🔧 ATTEMPTING TARGETED FIXES...")
    
    # Fix 1: Clean past_key_values issue
    try_fix_past_key_values(model, inputs, tokenizer)
    
    # Fix 2: Clean attention inputs issue  
    try_fix_attention_inputs(model, inputs, tokenizer)
    
    # Fix 3: Clean hidden states issue
    try_fix_hidden_states(model, inputs, tokenizer)

def try_fix_past_key_values(model, inputs, tokenizer):
    """Try to fix past_key_values string issue."""
    print(f"\n🔧 Fix 1: Attempting past_key_values fix...")
    
    try:
        # Patch the _extract_past_from_model_output method more aggressively
        if hasattr(model, '_extract_past_from_model_output'):
            original_extract_past = model._extract_past_from_model_output
            
            def fixed_extract_past(outputs, standardize_cache_format=None, **kwargs):
                print(f"   🔧 Intercepting _extract_past_from_model_output")
                
                # Always return None to disable past_key_values completely
                print(f"   🔧 Returning None to disable past_key_values")
                return None
            
            model._extract_past_from_model_output = fixed_extract_past
            
            # Test generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    use_cache=False  # Explicitly disable cache
                )
            print(f"   ✅ past_key_values fix successful!")
            return True
            
    except Exception as e:
        print(f"   ❌ past_key_values fix failed: {e}")
        return False

def try_fix_attention_inputs(model, inputs, tokenizer):
    """Try to fix attention inputs string issue."""
    print(f"\n🔧 Fix 2: Attempting attention inputs fix...")
    
    try:
        # Patch attention mechanism to clean string inputs
        def patch_attention_forward(original_forward):
            def cleaned_forward(*args, **kwargs):
                print(f"   🔧 Intercepting attention forward")
                
                # Clean all tensor-like arguments
                cleaned_args = []
                for arg in args:
                    if isinstance(arg, str):
                        print(f"   ⚠️  Filtering out string arg: {repr(arg[:50])}...")
                        continue
                    cleaned_args.append(arg)
                
                cleaned_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        print(f"   ⚠️  Filtering out string kwarg {key}: {repr(value[:50])}...")
                        continue
                    cleaned_kwargs[key] = value
                
                return original_forward(*cleaned_args, **cleaned_kwargs)
            
            return cleaned_forward
        
        # Find and patch attention modules
        attention_modules_found = 0
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'forward'):
                    print(f"   🔧 Patching attention module: {name}")
                    module.forward = patch_attention_forward(module.forward)
                    attention_modules_found += 1
        
        print(f"   🔧 Patched {attention_modules_found} attention modules")
        
        # Test generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False
            )
        print(f"   ✅ attention fix successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ attention fix failed: {e}")
        return False

def try_fix_hidden_states(model, inputs, tokenizer):
    """Try to fix hidden states string issue."""
    print(f"\n🔧 Fix 3: Attempting hidden states fix...")
    
    try:
        # Create a comprehensive tensor validator
        def validate_and_clean_tensor_input(value, name=""):
            if isinstance(value, str):
                print(f"   ⚠️  Found string in {name}: {repr(value[:50])}...")
                return None  # Filter out strings
            elif isinstance(value, (list, tuple)):
                cleaned = []
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        print(f"   ⚠️  Found string in {name}[{i}]: {repr(item[:50])}...")
                        continue  # Skip strings
                    cleaned.append(item)
                return cleaned if cleaned else None
            else:
                return value
        
        # Patch the main model forward method
        if hasattr(model.model, 'forward'):
            original_model_forward = model.model.forward
            
            def cleaned_model_forward(*args, **kwargs):
                print(f"   🔧 Intercepting model.forward")
                
                # Clean all inputs
                cleaned_args = []
                for i, arg in enumerate(args):
                    cleaned_arg = validate_and_clean_tensor_input(arg, f"arg[{i}]")
                    if cleaned_arg is not None:
                        cleaned_args.append(cleaned_arg)
                
                cleaned_kwargs = {}
                for key, value in kwargs.items():
                    cleaned_value = validate_and_clean_tensor_input(value, f"kwarg[{key}]")
                    if cleaned_value is not None:
                        cleaned_kwargs[key] = cleaned_value
                
                return original_model_forward(*cleaned_args, **cleaned_kwargs)
            
            model.model.forward = cleaned_model_forward
        
        # Test generation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False
            )
        print(f"   ✅ hidden states fix successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ hidden states fix failed: {e}")
        return False

def create_comprehensive_fix(model):
    """Create a comprehensive fix for all known CogVLM string/tensor issues."""
    print(f"\n🔧 APPLYING COMPREHENSIVE COGVLM FIX...")
    
    # Fix 1: Disable past_key_values completely
    if hasattr(model, '_extract_past_from_model_output'):
        def no_past_extract(outputs, **kwargs):
            return None
        model._extract_past_from_model_output = no_past_extract
        print(f"   ✅ Disabled past_key_values extraction")
    
    # Fix 2: Clean prepare_inputs_for_generation
    if hasattr(model, 'prepare_inputs_for_generation'):
        original_prepare = model.prepare_inputs_for_generation
        
        def cleaned_prepare_inputs(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            
            # Clean the result
            cleaned_result = {}
            for key, value in result.items():
                if isinstance(value, str):
                    print(f"   🔧 Filtering string from prepare_inputs: {key}")
                    continue
                elif isinstance(value, (list, tuple)):
                    cleaned_list = [item for item in value if not isinstance(item, str)]
                    if cleaned_list:
                        cleaned_result[key] = cleaned_list
                else:
                    cleaned_result[key] = value
            
            return cleaned_result
        
        model.prepare_inputs_for_generation = cleaned_prepare_inputs
        print(f"   ✅ Cleaned prepare_inputs_for_generation")
    
    # Fix 3: Override generate method with string filtering
    if hasattr(model, 'generate'):
        original_generate = model.generate
        
        def filtered_generate(*args, **kwargs):
            # Remove use_cache to prevent caching issues
            kwargs['use_cache'] = False
            kwargs['past_key_values'] = None
            
            print(f"   🔧 Calling generate with cleaned parameters")
            return original_generate(*args, **kwargs)
        
        model.generate = filtered_generate
        print(f"   ✅ Enhanced generate method")
    
    print(f"   🎉 Comprehensive fix applied!")

def test_comprehensive_fix():
    """Test the comprehensive fix."""
    print("\n=== TESTING COMPREHENSIVE FIX ===")
    
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
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Apply comprehensive fix
        create_comprehensive_fix(model)
        
        # Test with image
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        
        query = '<EOI>Question: What do you see in this image? Answer:'
        
        inputs = model.build_conversation_input_ids(
            tokenizer, 
            query=query, 
            history=[], 
            images=[image]
        )
        
        cleaned_inputs = clean_inputs_with_debug(inputs, device)
        
        print(f"\n🧪 Testing fixed generation...")
        
        with torch.no_grad():
            outputs = model.generate(
                **cleaned_inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        print(f"✅ SUCCESS! Generated outputs: {outputs.shape}")
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive fix test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run deep debug first
    deep_debug_cogvlm()
    
    print(f"\n" + "="*80)
    
    # Test comprehensive fix
    if test_comprehensive_fix():
        print(f"\n🎉 COMPREHENSIVE FIX SUCCESSFUL!")
        print(f"This fix can be integrated into the main script.")
    else:
        print(f"\n❌ Need further investigation...") 