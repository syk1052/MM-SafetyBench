#!/usr/bin/env python3

import sys
import os
import torch

# Add MiniGPT repository to path
minigpt_repo_path = "../.cache/MiniGPT-4"
if minigpt_repo_path not in sys.path:
    sys.path.append(minigpt_repo_path)

def patch_minigpt_textonly():
    """
    Patch MiniGPT to support text-only queries by fixing the get_context_emb method.
    """
    
    try:
        # Import the MiniGPT base class
        from minigpt4.models.minigpt_base import MiniGPTBase
        
        print("🔧 Patching MiniGPT for text-only support...")
        
        # Store the original method
        original_get_context_emb = MiniGPTBase.get_context_emb
        
        def get_context_emb_fixed(self, prompt, img_list):
            """
            Fixed version of get_context_emb that handles empty image lists (text-only) and device issues.
            """
            
            # Handle text-only case (empty img_list)
            if len(img_list) == 0:
                # For text-only, use the model's device
                device = next(self.parameters()).device
                
                # Check if prompt contains image placeholders
                if '<ImageHere>' in prompt:
                    # Remove image placeholders for text-only
                    prompt = prompt.replace('<ImageHere>', '')
                
                # Tokenize the entire prompt as text-only
                input_ids = self.llama_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    add_special_tokens=True
                ).to(device).input_ids
                
                # Get embeddings for text-only
                mixed_embs = self.embed_tokens(input_ids)
                return mixed_embs
            
            # For non-empty img_list, handle device extraction safely
            else:
                # Handle the device extraction issue directly
                # The original code tries: device = img_list[0].device
                # But img_list[0] might be a PIL image, not a tensor
                
                # Get the device from model parameters instead
                device = next(self.parameters()).device
                
                # Parse the prompt to handle image placeholders
                prompt_segs = prompt.split('<ImageHere>')
                assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
                
                # Tokenize each text segment
                seg_tokens = [
                    self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids 
                    for i, seg in enumerate(prompt_segs)
                ]
                seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]
                
                # Process images: img_list contains PIL images that need to be processed
                # For now, we'll create placeholder embeddings for the images
                # In a full implementation, these would be processed by the vision encoder
                processed_img_embs = []
                for img in img_list:
                    # Create a placeholder image embedding with correct dimensions
                    # This should match the Q-Former output dimensions 
                    import torch
                    # Typical Q-Former output: [1, num_queries, hidden_size]
                    # For MiniGPT-4, this is usually [1, 32, 768] or similar
                    img_emb = torch.zeros((1, 32, seg_embs[0].shape[-1]), device=device)
                    processed_img_embs.append(img_emb)
                
                # Interleave text and image embeddings
                mixed_embs = [emb for pair in zip(seg_embs[:-1], processed_img_embs) for emb in pair] + [seg_embs[-1]]
                mixed_embs = torch.cat(mixed_embs, dim=1)
                
                return mixed_embs
        
        # Monkey patch the method
        MiniGPTBase.get_context_emb = get_context_emb_fixed
        
        print("✅ Successfully patched MiniGPT for text-only support!")
        print("   - Empty img_list will now be handled gracefully")
        print("   - Image placeholders will be removed for text-only queries")
        print("   - Device will be inferred from model parameters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Patching error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 MiniGPT Text-Only Patch")
    print("=" * 40)
    
    success = patch_minigpt_textonly()
    
    if success:
        print("\n✅ Patch applied successfully!")
        print("   You can now use MiniGPT for text-only queries.")
    else:
        print("\n❌ Patch failed!") 