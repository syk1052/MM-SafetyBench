#!/usr/bin/env python3
"""
Debug script to exactly replicate the main script's CogVLM usage path
"""

import torch
from PIL import Image
import traceback
from transformers import LlamaTokenizer, AutoModelForCausalLM
import json
from pathlib import Path

def debug_main_script_path():
    """Debug the exact same path as the main script."""
    print("=== DEBUGGING MAIN SCRIPT'S COGVLM PATH ===")
    
    try:
        # Load model exactly as main script does
        print("1. Loading model exactly as main script...")
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
        
        # Load the exact same image as main script
        print("\n2. Loading exact same image as main script...")
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        print(f"Image: {image.size} - {image.mode}")
        
        # Load question exactly as main script does
        print("\n3. Loading question data...")
        questions_file = "../data/MM-SafetyBench/processed_questions/01-Illegal_Activitiy.json"
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        question_data = questions_data['0']  # Get first question
        question = question_data["Rephrased Question(SD)"]  # kind1 question
        print(f"Question: {question[:100]}...")
        
        # Build conversation inputs exactly as main script
        print("\n4. Building conversation inputs exactly as main script...")
        query = f'<EOI>Question: {question} Answer:'
        print(f"Query format: {query[:50]}...")
        
        inputs = model.build_conversation_input_ids(
            tokenizer, 
            query=query, 
            history=[], 
            images=[image]
        )
        
        print(f"Raw inputs keys: {list(inputs.keys())}")
        
        # Process inputs exactly as main script
        print("\n5. Processing inputs exactly as main script...")
        
        # Add batch dimension to tensors if they are 1D (main script fix)
        for key in ['input_ids', 'token_type_ids', 'attention_mask']:
            if key in inputs and torch.is_tensor(inputs[key]):
                original_shape = inputs[key].shape
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key].unsqueeze(0)
                    print(f"  {key}: {original_shape} -> {inputs[key].shape}")
                else:
                    print(f"  {key}: {inputs[key].shape} (no change)")
        
        # Move to device exactly as main script
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Convert dtypes exactly as main script
        if device == "cuda":
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.dtype == torch.float32:
                    original_dtype = v.dtype
                    inputs[k] = v.to(torch.bfloat16)
                    print(f"  {k}: {original_dtype} -> {inputs[k].dtype}")
                elif k == 'images' and isinstance(v, list):
                    for i, img_tensor in enumerate(v):
                        if torch.is_tensor(img_tensor) and img_tensor.dtype == torch.float32:
                            original_dtype = img_tensor.dtype
                            v[i] = img_tensor.to(device).to(torch.bfloat16)
                            print(f"  images[{i}]: {original_dtype} -> {v[i].dtype}")
        
        print(f"\n6. Final inputs inspection:")
        for k, v in inputs.items():
            if torch.is_tensor(v):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            elif k == 'images' and isinstance(v, list):
                for i, img in enumerate(v):
                    print(f"  images[{i}]: shape={img.shape}, dtype={img.dtype}, device={img.device}")
            else:
                print(f"  {k}: {type(v)}")
        
        # Test model.generate exactly as main script
        print(f"\n7. Testing model.generate() exactly as main script...")
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,  # Small for testing
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            print(f"✅ SUCCESS! Outputs shape: {outputs.shape}")
            
            # Test decoding
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response preview: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ FAILED in model.generate(): {e}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Let's debug step by step exactly what happens during generation
            print(f"\n8. Step-by-step debugging of generation process...")
            
            # Try to call the model forward directly
            try:
                print("Testing direct model forward...")
                with torch.no_grad():
                    model_outputs = model(**inputs, return_dict=True)
                print(f"✅ Direct model forward succeeded")
                
            except Exception as model_e:
                print(f"❌ Direct model forward failed: {model_e}")
                print("Traceback:")
                traceback.print_exc()
                
                # Debug the vision encoding specifically
                print(f"\n9. Debugging vision encoding specifically...")
                try:
                    print("Testing model.encode_images...")
                    images = inputs['images']
                    print(f"Images to encode: {len(images)} images")
                    for i, img in enumerate(images):
                        print(f"  Image {i}: {img.shape}, {img.dtype}, {img.device}")
                    
                    # Test the vision encoding path that's failing
                    images_features = model.model.encode_images(images)
                    print(f"✅ encode_images succeeded: {images_features.shape}")
                    
                except Exception as vision_e:
                    print(f"❌ encode_images failed: {vision_e}")
                    print("Traceback:")
                    traceback.print_exc()
                    
                    # Debug even deeper - test the vision model directly
                    print(f"\n10. Testing vision model directly...")
                    try:
                        vision_model = model.model.vision
                        image_tensor = images[0]
                        print(f"Testing vision model with: {image_tensor.shape}")
                        
                        if image_tensor.dim() == 3:
                            image_tensor = image_tensor.unsqueeze(0)
                            print(f"Added batch dim: {image_tensor.shape}")
                        
                        vision_output = vision_model(image_tensor)
                        print(f"✅ Direct vision model succeeded: {vision_output.shape}")
                        
                    except Exception as direct_vision_e:
                        print(f"❌ Direct vision model failed: {direct_vision_e}")
                        print("Traceback:")
                        traceback.print_exc()
                        
                        # Let's debug the patch embedding step by step
                        print(f"\n11. Debugging patch embedding step by step...")
                        debug_patch_embedding_issue(vision_model, image_tensor)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()

def debug_patch_embedding_issue(vision_model, image_tensor):
    """Debug the specific patch embedding issue."""
    try:
        print("=== DEBUGGING PATCH EMBEDDING ISSUE ===")
        
        # Get the patch embedding module
        patch_embedding = vision_model.patch_embedding
        print(f"Patch embedding type: {type(patch_embedding)}")
        
        # Inspect the patch embedding internals
        print(f"Patch embedding attributes:")
        for name, module in patch_embedding.named_children():
            print(f"  {name}: {type(module)}")
            if hasattr(module, 'weight'):
                print(f"    weight shape: {module.weight.shape}")
        
        # Test the Conv2d projection
        if hasattr(patch_embedding, 'proj'):
            proj = patch_embedding.proj
            print(f"\nTesting Conv2d projection...")
            print(f"  Input channels: {proj.in_channels}")
            print(f"  Output channels: {proj.out_channels}")
            print(f"  Kernel size: {proj.kernel_size}")
            print(f"  Stride: {proj.stride}")
            
            # Test the projection step
            x = proj(image_tensor)
            print(f"✅ Conv2d projection: {image_tensor.shape} -> {x.shape}")
            
            # Reshape for transformer
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).transpose(1, 2)
            print(f"✅ Reshape: -> {x.shape}")
            
            # Test position embedding
            if hasattr(patch_embedding, 'position_embedding'):
                pos_emb = patch_embedding.position_embedding
                print(f"Position embedding shape: {pos_emb.weight.shape}")
                
                if x.shape[1] != pos_emb.weight.shape[0]:
                    print(f"❌ MISMATCH: x.shape[1]={x.shape[1]} != pos_emb.shape[0]={pos_emb.weight.shape[0]}")
                    
                    # Check what the expected number of positions should be
                    expected_patches = (image_tensor.shape[2] // proj.stride[0]) * (image_tensor.shape[3] // proj.stride[1])
                    print(f"Expected patches: {expected_patches}")
                    print(f"Position embeddings available: {pos_emb.weight.shape[0]}")
                    
                    # The issue might be here - let's see what happens with different image sizes
                    print(f"Testing different image sizes...")
                    for size in [224, 336, 448, 490, 512]:
                        patches = (size // proj.stride[0]) * (size // proj.stride[1])
                        print(f"  Size {size}x{size}: {patches} patches")
                        if patches == pos_emb.weight.shape[0] - 1:  # -1 for CLS token
                            print(f"    ✅ MATCH! Expected size: {size}x{size}")
                        elif patches == pos_emb.weight.shape[0]:
                            print(f"    ✅ MATCH (with CLS)! Expected size: {size}x{size}")
            
            # Test CLS token
            if hasattr(patch_embedding, 'cls_embedding'):
                cls_emb = patch_embedding.cls_embedding
                print(f"CLS embedding shape: {cls_emb.shape}")
                
                # Test concatenation
                batch_size = x.shape[0]
                cls_expanded = cls_emb.expand(batch_size, -1, -1)
                print(f"CLS expanded: {cls_expanded.shape}")
                
                print(f"Attempting concatenation:")
                print(f"  cls: {cls_expanded.shape}")
                print(f"  x: {x.shape}")
                
                combined = torch.cat((cls_expanded, x), dim=1)
                print(f"✅ Concatenation successful: {combined.shape}")
                
    except Exception as e:
        print(f"❌ Error in patch embedding debug: {e}")
        traceback.print_exc()

def test_image_preprocessing_differences():
    """Test if there are differences in how images are preprocessed."""
    print("\n=== TESTING IMAGE PREPROCESSING DIFFERENCES ===")
    
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
        
        # Test the same image with our debug approach vs main script approach
        image_path = "../data/MM-SafetyBench/imgs/01-Illegal_Activitiy/SD/0.jpg"
        image = Image.open(image_path).convert('RGB')
        
        print(f"Original image size: {image.size}")
        
        # Method 1: Debug script approach (working)
        print(f"\n1. Debug script approach (working):")
        query1 = '<EOI>Question: What do you see in this image? Answer:'
        inputs1 = model.build_conversation_input_ids(tokenizer, query=query1, history=[], images=[image])
        img_tensor1 = inputs1['images'][0]
        print(f"  Result: {img_tensor1.shape}")
        
        # Method 2: Main script approach (failing)
        print(f"\n2. Main script approach (failing):")
        
        # Load actual question from dataset
        questions_file = "../data/MM-SafetyBench/processed_questions/01-Illegal_Activitiy.json"
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        question = questions_data['0']["Rephrased Question(SD)"]
        
        query2 = f'<EOI>Question: {question} Answer:'
        inputs2 = model.build_conversation_input_ids(tokenizer, query=query2, history=[], images=[image])
        img_tensor2 = inputs2['images'][0]
        print(f"  Result: {img_tensor2.shape}")
        
        # Compare the two
        print(f"\n3. Comparison:")
        print(f"  Debug query length: {len(query1)}")
        print(f"  Main query length: {len(query2)}")
        print(f"  Same image tensor: {torch.equal(img_tensor1, img_tensor2)}")
        print(f"  Tensor shape match: {img_tensor1.shape == img_tensor2.shape}")
        
        if img_tensor1.shape != img_tensor2.shape:
            print(f"  ❌ TENSOR SHAPES DIFFER!")
            print(f"    Debug: {img_tensor1.shape}")
            print(f"    Main: {img_tensor2.shape}")
        else:
            print(f"  ✅ Tensor shapes match - issue is elsewhere")
        
    except Exception as e:
        print(f"❌ Error in preprocessing test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_main_script_path()
    test_image_preprocessing_differences() 