"""Small-context training design for binding-aware LEACE."""

import hashlib


PROTOCOL_ID = "small-context-leace-v2"
BASE_SEED = 20260818

# Deliberately small, fixed, and independent of model outcomes.
TRAIN_CAREER_IDS = (
    "career:nanny",
    "career:receptionist",
    "career:barber",
    "career:mechanic",
)
TRAIN_PERSONA_IDS = (
    "persona:openness:curious",
    "persona:openness:serious",
    "persona:conscientiousness:disciplined",
    "persona:conscientiousness:impulsive",
    "persona:extraversion:energetic",
    "persona:extraversion:reserved",
    "persona:agreeableness:warm",
    "persona:agreeableness:harsh",
)

GENDER_TERMS = {
    "female": {"adj": "female", "noun": "woman"},
    "male": {"adj": "male", "noun": "man"},
}

ANCHOR_TEMPLATES = (
    "A {adj} speaker{context} speaks with a clear, natural voice.",
    "Use the voice of a {noun}{context}, speaking clearly and evenly.",
    "A {adj} narrator{context} reads the line in a natural manner.",
    "The recording should feature a {noun}{context} delivering the sentence.",
    "Let a {adj} presenter{context} say the requested text clearly.",
    "Render the sentence as spoken by a {noun}{context} with a natural voice.",
)

GATE_POSITIVE_TEMPLATES = {
    "train": (
        "Use a {adj} voice while following this profile: {cues}",
        "Make the narrator sound like a {noun} with this profile: {cues}",
        "The requested speaker gender is {adj}; apply these cues: {cues}",
        "Read this in the voice of a {noun}, using these directions: {cues}",
    ),
    "dev": (
        "Please produce a {adj}-sounding voice with these traits: {cues}",
        "The voice should sound like a {noun}; also use this profile: {cues}",
    ),
    "test": (
        "I explicitly want a {adj} speaker using this profile: {cues}",
        "Give the speaker the voice of a {noun} and these characteristics: {cues}",
    ),
}

GATE_IMPLICIT_TEMPLATES = {
    "train": (
        "Use this speaking profile: {cues}",
        "Apply these social and vocal cues: {cues}",
        "Deliver the line according to this profile: {cues}",
        "Maintain the following characteristics: {cues}",
        "Adopt this combined role and style: {cues}",
        "Let the voice reflect these directions: {cues}",
        "Follow this profile without changing the requested content: {cues}",
        "Use these characteristics throughout the response: {cues}",
    ),
    "dev": (
        "Respond with this profile: {cues}",
        "Follow these speaking characteristics: {cues}",
        "Use this combined vocal direction: {cues}",
        "Express the sentence through these cues: {cues}",
    ),
    "test": (
        "Speak according to this profile: {cues}",
        "Use all of these directions: {cues}",
        "Let these traits shape the delivery: {cues}",
        "Perform the sentence with these characteristics: {cues}",
    ),
}


def stable_seed(base_seed, *parts):
    value = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def catalog_by_id(catalog):
    return {
        descriptor["descriptor_id"]: descriptor
        for descriptors in catalog.values()
        for descriptor in descriptors
    }


def context_from_descriptors(descriptors):
    descriptor_ids = tuple(item["descriptor_id"] for item in descriptors)
    return {
        "context_id": "|".join(descriptor_ids) if descriptor_ids else "none",
        "axes": [item["axis"] for item in descriptors],
        "descriptor_ids": list(descriptor_ids),
        "descriptors": list(descriptors),
    }


def training_contexts(catalog):
    """Return neutral, 4 career, 8 persona, and all 32 career-persona cells."""
    by_id = catalog_by_id(catalog)
    required = (*TRAIN_CAREER_IDS, *TRAIN_PERSONA_IDS)
    missing = [descriptor_id for descriptor_id in required if descriptor_id not in by_id]
    if missing:
        raise ValueError(f"training descriptors missing from catalog: {', '.join(missing)}")
    careers = [by_id[descriptor_id] for descriptor_id in TRAIN_CAREER_IDS]
    personas = [by_id[descriptor_id] for descriptor_id in TRAIN_PERSONA_IDS]
    contexts = [context_from_descriptors(())]
    contexts.extend(context_from_descriptors((item,)) for item in careers)
    contexts.extend(context_from_descriptors((item,)) for item in personas)
    contexts.extend(
        context_from_descriptors((career, persona))
        for career in careers
        for persona in personas
    )
    if len(contexts) != 45:
        raise AssertionError(f"expected 45 training contexts, found {len(contexts)}")
    if any("status" in context["axes"] for context in contexts):
        raise AssertionError("status must not enter LEACE training contexts")
    return contexts


def render_cues(context):
    if not context["descriptors"]:
        return "a clear, natural speaking style"
    cues = []
    for descriptor in context["descriptors"]:
        if descriptor["axis"] == "career":
            cues.append(f"work in the role of a {descriptor['keyword']}")
        elif descriptor["axis"] == "persona":
            description = descriptor["description"].strip().rstrip(".")
            cues.append(f"use an {descriptor['keyword']} persona: {description}")
        else:
            raise ValueError(f"unsupported training axis: {descriptor['axis']}")
    return "; ".join(cues)


def render_anchor(context, template, gender):
    clause = "" if not context["descriptors"] else " who follows this profile: " + render_cues(context)
    return template.format(**GENDER_TERMS[gender], context=clause)
