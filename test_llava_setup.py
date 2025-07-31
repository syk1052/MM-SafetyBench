#!/usr/bin/env python3
"""
Test script to verify LLaVA-1.5(7B) setup and basic functionality.
Run this before using add_llava_responses.py to ensure everything works.
"""

import sys
import torch
from add_model_responses import LlavaQueryEngine

def test_system_requirements():
    """Test system requirements and dependencies."""
    print("=== Testing System Requirements ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    return True

def test_llava_loading():
    """Test loading LLaVA model."""
    print("\n=== Testing LLaVA Model Loading ===")
    
    try:
        llava_engine = LlavaQueryEngine()
        print("✓ LlavaQueryEngine initialized")
        
        # Test loading (this will download the model if not cached)
        llava_engine.load_model()
        print("✓ LLaVA model loaded successfully")
        
        return llava_engine
    except Exception as e:
        print(f"✗ Error loading LLaVA: {e}")
        return None

def test_simple_query(llava_engine):
    """Test a simple query."""
    print("\n=== Testing Simple Query ===")
    
    if llava_engine is None:
        print("✗ Cannot test query - model not loaded")
        return False
    
    try:
        # Test with a safe question
        test_question = "What is the capital of France?"
        response = llava_engine.query_text_only(test_question)
        
        print(f"Question: {test_question}")
        print(f"Response: {response}")
        print("✓ Simple query successful")
        return True
        
    except Exception as e:
        print(f"✗ Error during query: {e}")
        return False

def test_image_query(llava_engine):
    """Test query with image."""
    print("\n=== Testing Image Query ===")
    
    if llava_engine is None:
        print("✗ Cannot test image query - model not loaded")
        return False
    
    try:
        # Test with image
        test_question = "The image shows a phrase. What do you see?"
        test_phrase = "Hello World"
        response = llava_engine.query_with_image(test_question, test_phrase)
        
        print(f"Question: {test_question}")
        print(f"Key phrase in image: {test_phrase}")
        print(f"Response: {response}")
        print("✓ Image query successful")
        return True
        
    except Exception as e:
        print(f"✗ Error during image query: {e}")
        return False

def main():
    """Run all tests."""
    print("LLaVA-1.5(7B) Setup Test")
    print("=" * 50)
    
    # Test system requirements
    if not test_system_requirements():
        print("System requirements test failed!")
        return False
    
    # Test model loading
    llava_engine = test_llava_loading()
    
    # Test simple query
    if not test_simple_query(llava_engine):
        print("Simple query test failed!")
        return False
    
    # Test image query
    if not test_image_query(llava_engine):
        print("Image query test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! LLaVA is ready to use.")
    print("You can now run: python add_llava_responses.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 