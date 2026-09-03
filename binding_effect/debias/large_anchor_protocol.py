"""High-coverage, counterfactual anchor design for Parler Large."""

import hashlib

from debias.binding_protocol import context_from_descriptors, render_cues


PROTOCOL_ID = "parler-large-balanced-anchors-v1"
BASE_SEED = 20260828

# Adult gender terms only. Each template differs solely at one marked term.
TRAIN_TEMPLATES = (
    ("A female speaker{context} speaks with a clear, natural voice.",
     "A male speaker{context} speaks with a clear, natural voice."),
    ("Use the voice of a woman{context}, speaking clearly and evenly.",
     "Use the voice of a man{context}, speaking clearly and evenly."),
    ("A female narrator{context} reads the line in a natural manner.",
     "A male narrator{context} reads the line in a natural manner."),
    ("The recording should feature a woman{context} delivering the sentence.",
     "The recording should feature a man{context} delivering the sentence."),
    ("Let a female presenter{context} say the requested text clearly.",
     "Let a male presenter{context} say the requested text clearly."),
    ("Render the sentence as spoken by a woman{context} with a natural voice.",
     "Render the sentence as spoken by a man{context} with a natural voice."),
    ("Please assign a female reader{context} to deliver this line.",
     "Please assign a male reader{context} to deliver this line."),
    ("The requested voice is that of a woman{context}; keep the delivery steady.",
     "The requested voice is that of a man{context}; keep the delivery steady."),
    ("Have a female host{context} speak the supplied sentence.",
     "Have a male host{context} speak the supplied sentence."),
    ("Produce this utterance with a woman's voice{context}.",
     "Produce this utterance with a man's voice{context}."),
    ("For this recording, choose a female announcer{context}.",
     "For this recording, choose a male announcer{context}."),
    ("The line is to be read by a woman{context} in a natural tone.",
     "The line is to be read by a man{context} in a natural tone."),
    ("Generate a female-sounding adult speaker{context} for this sentence.",
     "Generate a male-sounding adult speaker{context} for this sentence."),
    ("An adult woman{context} should voice the requested content.",
     "An adult man{context} should voice the requested content."),
)


def _stable_rank(*parts):
    value = "\x1f".join([str(BASE_SEED), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expanded_contexts(catalog):
    """Return 180 status-free contexts with full univariate descriptor coverage."""
    careers = sorted(catalog["career"], key=lambda row: row["descriptor_id"])
    personas = sorted(catalog["persona"], key=lambda row: row["descriptor_id"])
    contexts = [context_from_descriptors(())]
    contexts.extend(context_from_descriptors((row,)) for row in careers)
    contexts.extend(context_from_descriptors((row,)) for row in personas)

    # Seed the composite inventory so every descriptor occurs at least once.
    required = {(index % len(careers), index) for index in range(len(personas))}
    candidates = [
        (career_index, persona_index)
        for career_index in range(len(careers))
        for persona_index in range(len(personas))
        if (career_index, persona_index) not in required
    ]
    candidates.sort(key=lambda pair: _stable_rank(
        careers[pair[0]]["descriptor_id"], personas[pair[1]]["descriptor_id"],
    ))
    selected = sorted(required | set(candidates[:112 - len(required)]))
    contexts.extend(
        context_from_descriptors((careers[career_index], personas[persona_index]))
        for career_index, persona_index in selected
    )
    if len(contexts) != 180:
        raise AssertionError(f"expected 180 contexts, found {len(contexts)}")
    return contexts


def render_large_anchor(context, template_pair, gender):
    clause = "" if not context["descriptors"] else " who follows this profile: " + render_cues(context)
    return template_pair[gender == "male"].format(context=clause)
