#!/usr/bin/env python3
"""
Simple test to try different LLaVA approaches and find one that works.
"""

import torch
from PIL import Image, ImageDraw, ImageFont

def test_llava_approach_1():
    """Test with LlavaProcessor instead of LlavaNextProcessor"""
    print("=== Testing LlavaProcessor approach ===")
    
    try:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        
        model_name = "llava-hf/llava-1.5-7b-hf"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading with LlavaProcessor on {device}...")
        processor = LlavaProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Create a simple test image
        image = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(image)
        text = "test phrase"
        draw.text((50, 100), text, fill='black')
        
        # Simple test prompt
        prompt = "USER: <image>\nWhat do you see in this image?\nASSISTANT:"
        
        print("Processing inputs...")
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        print("Response:", response)
        
        return True, "LlavaProcessor approach works"
        
    except Exception as e:
        print(f"LlavaProcessor approach failed: {e}")
        return False, str(e)

def test_llava_approach_2():
    """Test text-only approach"""
    print("\n=== Testing text-only approach ===")
    
    try:
        from transformers import LlavaProcessor, LlavaForConditionalGeneration
        
        model_name = "llava-hf/llava-1.5-7b-hf"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading for text-only on {device}...")
        processor = LlavaProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Simple text prompt without image tags
        prompt = "USER: What is the capital of France?\nASSISTANT:"
        
        print("Processing text-only inputs...")
        # Try to process without image
        inputs = processor(text=prompt, images=None, return_tensors="pt").to(device)
        
        print("Generating response...")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        print("Response:", response)
        
        return True, "Text-only approach works"
        
    except Exception as e:
        print(f"Text-only approach failed: {e}")
        return False, str(e)

def main():
    print("Testing different LLaVA approaches to find a working solution...")
    
    # Test approach 1
    success1, msg1 = test_llava_approach_1()
    
    # Test approach 2  
    success2, msg2 = test_llava_approach_2()
    
    print(f"\n=== Results ===")
    print(f"Approach 1 (with image): {'✓' if success1 else '✗'} - {msg1}")
    print(f"Approach 2 (text-only): {'✓' if success2 else '✗'} - {msg2}")
    
    if success1 or success2:
        print("\n✓ Found working approach!")
        return True
    else:
        print("\n✗ No working approach found")
        return False

if __name__ == "__main__":
    main() 