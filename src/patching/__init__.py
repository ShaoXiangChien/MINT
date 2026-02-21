from .hooks import register_capture_hook, register_patch_hook, remove_hooks
from .decoder_patching import capture_decoder_hs, patch_decoder_and_generate
from .vision_patching import capture_vision_emb, patch_vision_and_generate
