"""Parler-TTS activation extraction and generation integration."""

import re

import torch


_EXPLICIT_GENDER = re.compile(
    r"\b(?:female|male|woman|man|women|men|girl|boy|feminine|masculine)\b",
    flags=re.IGNORECASE,
)


def has_explicit_gender_command(description):
    """Return whether a description directly requests a gendered voice."""
    return bool(_EXPLICIT_GENDER.search(description or ""))


def masked_mean(hidden_states, attention_mask):
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("expected hidden states [batch, tokens, features] and mask [batch, tokens]")
    weights = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1).clamp_min(1)
    return (hidden_states * weights).sum(dim=1) / counts


def encode_description_states(model, input_ids, attention_mask):
    """Compute the projected description states consumed by Parler's decoder."""
    outputs = model.get_text_encoder()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    encoder_config = model.get_text_encoder().config
    decoder_config = model.decoder.config
    if (
        encoder_config.hidden_size != decoder_config.hidden_size
        and decoder_config.cross_attention_hidden_size is None
    ):
        hidden_states = model.enc_to_dec_proj(hidden_states)
    return hidden_states * attention_mask[..., None].to(hidden_states.dtype)


def collect_pooled_activations(model, tokenizer, texts, *, batch_size=16, device=None):
    """Encode text descriptions and return CPU float32 masked means."""
    if not texts:
        raise ValueError("at least one description is required")
    if device is None:
        device = next(model.parameters()).device
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            states = encode_description_states(model, input_ids, attention_mask)
            batches.append(masked_mean(states, attention_mask).float().cpu())
    return torch.cat(batches, dim=0)


def generate_with_eraser(
    model,
    eraser,
    description,
    input_ids,
    attention_mask,
    *,
    bypass_explicit=True,
    **generation_kwargs,
):
    """Generate from precomputed encoder states, optionally erased by LEACE."""
    from transformers.modeling_outputs import BaseModelOutput

    states = encode_description_states(model, input_ids, attention_mask)
    if not (bypass_explicit and has_explicit_gender_command(description)):
        states = eraser(states)
        states = states * attention_mask[..., None].to(states.dtype)
    return model.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=states),
        attention_mask=attention_mask,
        **generation_kwargs,
    )
