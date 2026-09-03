"""VoxInstruct projected-state extraction and constant steering helpers."""

import torch

from debias.parler import has_explicit_gender_command, masked_mean


REPRESENTATIONS = {
    "ar": "voxinstruct_ar_projected_mt5_masked_mean",
    "nar": "voxinstruct_nar_projected_mt5_masked_mean",
}


def format_instruction(description, prompt_text=None):
    """Match the text normalization used by VoxInstruct generation."""
    text = description if not prompt_text else f'{description}. "{prompt_text}"'
    return text.strip().capitalize()


def steer_text_states(states, attention_mask, steering, strength):
    """Apply a constant pooled-state offset while preserving padded zeros."""
    if not strength >= 0.0:
        raise ValueError("intervention_strength must be non-negative")
    pooled = masked_mean(states, attention_mask)
    shifted = states + strength * (steering(pooled) - pooled).unsqueeze(1)
    return shifted * attention_mask[..., None].to(shifted.dtype)


def initialize_nar_sequences(seqs, st_lens, codebooks):
    """Copy AR tokens only where the NAR decoder has known inputs."""
    full_seqs = torch.stack([seqs] * codebooks, dim=1)
    positions = torch.arange(seqs.shape[1], device=seqs.device)
    known_codebook = torch.arange(codebooks, device=seqs.device)[None, :] < 1
    semantic_prefix = positions[None, :] < st_lens[:, None]
    known = known_codebook[:, :, None] | semantic_prefix[:, None, :]
    return torch.where(known, full_seqs, torch.zeros_like(full_seqs))


def collect_pooled_activations(model, tokenizer, texts, *, max_length, batch_size=16,
                               device=None):
    """Encode Vox instructions and return CPU float32 projected masked means."""
    if not texts:
        raise ValueError("at least one instruction is required")
    if device is None:
        device = next(model.parameters()).device
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start:start + batch_size], padding="max_length", truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            states = model.encode_text(input_ids, attention_mask)
            batches.append(masked_mean(states.float(), attention_mask).cpu())
    return torch.cat(batches, dim=0)


__all__ = [
    "REPRESENTATIONS",
    "collect_pooled_activations",
    "format_instruction",
    "has_explicit_gender_command",
    "initialize_nar_sequences",
    "steer_text_states",
]
