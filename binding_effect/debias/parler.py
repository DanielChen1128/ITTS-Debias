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


def collect_pooled_layer_activations(model, tokenizer, texts, *, batch_size=16, device=None):
    """Return masked means for the embedding output and every encoder layer."""
    if not texts:
        raise ValueError("at least one description is required")
    if device is None:
        device = next(model.parameters()).device
    batches = None
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size], padding=True, truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            outputs = model.get_text_encoder()(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
            states = outputs.hidden_states
            if not states:
                raise ValueError("text encoder did not return hidden states")
            # T5-XL intermediate fp16 states can overflow while summing tokens.
            pooled = [masked_mean(state.float(), attention_mask).cpu() for state in states]
            if batches is None:
                batches = [[] for _ in pooled]
            if len(batches) != len(pooled):
                raise ValueError("text encoder returned an inconsistent layer count")
            for index, values in enumerate(pooled):
                batches[index].append(values)
    return [torch.cat(values, dim=0) for values in batches]


def erase_description_states(states, attention_mask, eraser, *, mode="token"):
    """Apply an eraser token-wise or as one pooled-mean shift per prompt."""
    if mode == "token":
        erased = eraser(states)
    elif mode == "pooled-shift":
        pooled = masked_mean(states, attention_mask)
        erased = states + (eraser(pooled) - pooled).unsqueeze(1)
    else:
        raise ValueError("intervention_mode must be 'token' or 'pooled-shift'")
    return erased * attention_mask[..., None].to(erased.dtype)


def generate_with_eraser(
    model,
    eraser,
    description,
    input_ids,
    attention_mask,
    *,
    bypass_explicit=True,
    intervention_strength=1.0,
    intervention_mode="token",
    return_intervention=False,
    **generation_kwargs,
):
    """Generate from precomputed encoder states, optionally erased by LEACE."""
    from transformers.modeling_outputs import BaseModelOutput

    states = encode_description_states(model, input_ids, attention_mask)
    bypassed = bypass_explicit and has_explicit_gender_command(description)
    bypass_mode = "regex" if bypass_explicit else "disabled"
    if not intervention_strength >= 0.0:
        raise ValueError("intervention_strength must be non-negative")
    if not bypassed:
        erased = erase_description_states(
            states, attention_mask, eraser, mode=intervention_mode,
        )
        states = states + intervention_strength * (erased - states)
        states = states * attention_mask[..., None].to(states.dtype)
    generation = model.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=states),
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    if not return_intervention:
        return generation
    return generation, {
        "bypass_mode": bypass_mode,
        "bypassed": bool(bypassed),
        "eraser_applied": not bypassed,
        "intervention_strength": intervention_strength,
        "intervention_mode": intervention_mode,
    }


def generate_with_eraser_batch(
    model, eraser, descriptions, input_ids, attention_mask, *, bypass_explicit=True,
    intervention_strength=1.0, intervention_mode="token",
    return_intervention=False,
    **generation_kwargs,
):
    """Batched counterpart that applies bypass decisions independently per row."""
    from transformers.modeling_outputs import BaseModelOutput

    states = encode_description_states(model, input_ids, attention_mask)
    bypassed = torch.tensor(
        [bypass_explicit and has_explicit_gender_command(text) for text in descriptions],
        device=states.device,
    )
    bypass_mode = "regex" if bypass_explicit else "disabled"
    if not intervention_strength >= 0.0:
        raise ValueError("intervention_strength must be non-negative")
    transformed = erase_description_states(
        states, attention_mask, eraser, mode=intervention_mode,
    )
    erased = states + intervention_strength * (transformed - states)
    states = torch.where(bypassed[:, None, None], states, erased)
    states = states * attention_mask[..., None].to(states.dtype)
    generation = model.generate(
        encoder_outputs=BaseModelOutput(last_hidden_state=states), attention_mask=attention_mask,
        **generation_kwargs,
    )
    if not return_intervention:
        return generation
    records = [
        {
            "bypass_mode": bypass_mode,
            "bypassed": bool(row_bypassed.item()),
            "eraser_applied": not bool(row_bypassed.item()),
            "intervention_strength": intervention_strength,
            "intervention_mode": intervention_mode,
        }
        for row_bypassed in bypassed
    ]
    return generation, records
