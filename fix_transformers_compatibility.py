#!/usr/bin/env python3

import sys
import os
import torch
from functools import wraps

# Add MiniGPT repository to path
minigpt_repo_path = "../.cache/MiniGPT-4"
if minigpt_repo_path not in sys.path:
    sys.path.append(minigpt_repo_path)

def patch_transformers_compatibility():
    """
    Patch transformers compatibility issues for newer versions with MiniGPT.
    Specifically fixes the 'cache_position' parameter issue.
    """
    
    try:
        print("🔧 Patching transformers compatibility...")
        
        # Import transformers components
        from transformers import LlamaForCausalLM
        
        # Store the original forward method
        original_forward = LlamaForCausalLM.forward
        
        @wraps(original_forward)
        def forward_compatible(self, *args, **kwargs):
            """
            Compatible forward method that filters out unsupported parameters.
            """
            # List of parameters that newer transformers might pass but older models don't support
            unsupported_params = [
                'cache_position',
                'position_embeddings', 
                'hard_labels',
                'num_logits_to_keep',
                'inputs_embeds',  # Sometimes conflicts
                'position_ids'    # Can cause issues in some versions
            ]
            
            # Filter out unsupported parameters
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in unsupported_params}
            
            # If any parameters were filtered, log it for debugging
            filtered_params = [k for k in kwargs.keys() if k in unsupported_params]
            if filtered_params:
                print(f"🔧 Filtered forward parameters: {filtered_params}")
            
            # Call the original forward method with filtered parameters
            try:
                return original_forward(self, *args, **filtered_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Extract the problematic parameter from error message
                    import re
                    match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                    if match:
                        param_name = match.group(1)
                        print(f"🔧 Removing additional problematic parameter: {param_name}")
                        # Remove the problematic parameter and try again
                        filtered_kwargs.pop(param_name, None)
                        return original_forward(self, *args, **filtered_kwargs)
                raise
        
        # Monkey patch the forward method
        LlamaForCausalLM.forward = forward_compatible
        
        # Also patch the __call__ method to catch parameters before they reach forward
        original_call = LlamaForCausalLM.__call__
        
        @wraps(original_call)
        def call_compatible(self, *args, **kwargs):
            """Compatible __call__ method that filters parameters before forward."""
            # Only filter the most problematic parameters that definitely cause issues
            problematic_params = ['cache_position']
            
            # Filter kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_params}
            
            # Log filtered parameters (only once per session to avoid spam)
            filtered_params = [k for k in kwargs.keys() if k in problematic_params]
            if filtered_params:
                if not hasattr(call_compatible, '_logged_filters'):
                    call_compatible._logged_filters = set()
                filter_key = tuple(sorted(filtered_params))
                if filter_key not in call_compatible._logged_filters:
                    print(f"🔧 Filtered __call__ parameters: {filtered_params}")
                    call_compatible._logged_filters.add(filter_key)
            
            try:
                return original_call(self, *args, **filtered_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Extract and remove any additional problematic parameter
                    import re
                    match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                    if match:
                        param_name = match.group(1)
                        print(f"🔧 Removing additional problematic __call__ parameter: {param_name}")
                        filtered_kwargs.pop(param_name, None)
                        return original_call(self, *args, **filtered_kwargs)
                raise
        
        LlamaForCausalLM.__call__ = call_compatible
        
        print("✅ Successfully patched transformers compatibility!")
        print("   - Filtered unsupported parameters: cache_position, etc.")
        print("   - LlamaForCausalLM.forward() now compatible with newer transformers")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Patching error: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_generation_utils():
    """
    Additional patch for generation utilities if needed.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        
        # Store original generate method
        if hasattr(GenerationMixin, '_original_generate'):
            # Already patched
            return True
            
        original_generate = GenerationMixin.generate
        GenerationMixin._original_generate = original_generate
        
        @wraps(original_generate)
        def generate_compatible(self, *args, **kwargs):
            """
            Compatible generate method that handles parameter filtering.
            """
            # Remove parameters that might cause issues
            problematic_params = [
                'cache_position', 
                'position_embeddings',
                'hard_labels',
                'num_logits_to_keep',
                'use_cache_quantization',
                'cache_implementation'
            ]
            
            # Filter kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in problematic_params}
            
            # Log if any parameters were filtered
            filtered_params = [k for k in kwargs.keys() if k in problematic_params]
            if filtered_params:
                print(f"🔧 Filtered generate parameters: {filtered_params}")
            
            try:
                return original_generate(self, *args, **filtered_kwargs)
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Extract and remove the problematic parameter
                    import re
                    match = re.search(r"unexpected keyword argument '(\w+)'", str(e))
                    if match:
                        param_name = match.group(1)
                        print(f"🔧 Removing additional problematic generate parameter: {param_name}")
                        filtered_kwargs.pop(param_name, None)
                        return original_generate(self, *args, **filtered_kwargs)
                raise
        
        GenerationMixin.generate = generate_compatible
        
        print("✅ Generation utilities patched successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Generation utils patch failed (non-critical): {e}")
        return False

def patch_internal_generation_methods():
    """
    Patch internal generation methods to handle cache_position parameter.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        
        # Patch _sample method if it exists
        if hasattr(GenerationMixin, '_sample'):
            original_sample = GenerationMixin._sample
            
            @wraps(original_sample)
            def _sample_compatible(self, *args, **kwargs):
                """Compatible _sample method with comprehensive parameter filtering."""
                import inspect
                
                # Get the actual _sample method to see what we're working with
                try:
                    # Call the original method but intercept if it fails
                    return original_sample(self, *args, **kwargs)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e):
                        print(f"🔧 _sample method failed with: {e}")
                        # Try calling with minimal parameters
                        filtered_kwargs = {k: v for k, v in kwargs.items() 
                                         if k not in ['cache_position', 'position_embeddings', 'hard_labels', 'num_logits_to_keep']}
                        return original_sample(self, *args, **filtered_kwargs)
                    raise
            
            GenerationMixin._sample = _sample_compatible
            
        # Also patch the model call within generation
        print("✅ Internal generation methods patched!")
        return True
        
    except Exception as e:
        print(f"⚠️ Internal generation patch failed (non-critical): {e}")
        return False

if __name__ == "__main__":
    print("🚀 Transformers Compatibility Patch")
    print("=" * 45)
    
    success1 = patch_transformers_compatibility()
    success2 = patch_generation_utils()
    success3 = patch_internal_generation_methods()
    
    if success1:
        print("\n✅ Main compatibility patch applied successfully!")
        print("   MiniGPT should now work with newer transformers versions.")
    else:
        print("\n❌ Main patch failed!")
    
    if success2:
        print("✅ Generation utils patch applied successfully!")
    
    if success3:
        print("✅ Internal generation methods patch applied successfully!")
    
    print("\n🔧 Ready to use with transformers", end="")
    try:
        import transformers
        print(f" {transformers.__version__}")
    except:
        print(" (version unknown)") 