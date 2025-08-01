#!/usr/bin/env python3
"""
Test script to verify CogVLM fixes work properly
"""

import sys
sys.path.append('.')

from add_multimodal_responses_optimized import OptimizedMultimodalQueryEngine

def test_cogvlm_fix():
    print("🧪 Testing CogVLM compatibility fixes...")
    
    try:
        # Initialize the model
        engine = OptimizedMultimodalQueryEngine("cogvlm-chat-17b")
        engine.load_model()
        
        # Test a simple text query
        test_question = "What is artificial intelligence?"
        print(f"\n📝 Testing question: {test_question}")
        
        response = engine.query_text_only(test_question, max_new_tokens=50)
        print(f"✅ Response received: {response[:100]}...")
        
        print("\n🎉 CogVLM test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_cogvlm_fix() 