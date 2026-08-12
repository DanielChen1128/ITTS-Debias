"""Dataset loading, keyword parsing, and leakage-safe split utilities.

The Binding Effect descriptor files share a schema of
``description / trait / keywords / prompt_text / id``. This module normalises
those records into axis-tagged descriptors and derives grouping metadata used
for evaluation. Stereotype gender priors are treated strictly as analysis
covariates; they are never used as LEACE fit labels.
"""

from pathlib import Path
import hashlib
import json

# The ten neutral carrier transcripts shared across every descriptor file.
TRANSCRIPTS = (
    "What\u2019s your plan for the weekend?",
    "It\u2019s great to see you again!",
    "How would you describe your current mood?",
    "Could you please introduce yourself briefly?",
    "How do you feel about today\u2019s weather?",
    "Hey, how are you doing today?",
    "Can you tell me something interesting about your work?",
    "What are your thoughts on this topic?",
    "Good morning, how\u2019s everything going?",
    "Let\u2019s start by talking about your day.",
)

CANONICAL_TRANSCRIPTS = (
    "Hey, how are you doing today?",
    "Good morning, how\u2019s everything going?",
    "Could you please introduce yourself briefly?",
    "What are your thoughts on this topic?",
    "Let\u2019s start by talking about your day.",
    "How do you feel about today\u2019s weather?",
    "Can you tell me something interesting about your work?",
    "It\u2019s great to see you again!",
    "How would you describe your current mood?",
    "What\u2019s your plan for the weekend?",
)

TRANSCRIPTS_BY_PROFILE = {
    "canonical-stage1-6900-v1": CANONICAL_TRANSCRIPTS,
    "legacy-5900-v1": TRANSCRIPTS,
}

# Career stereotype gender priors that are directly grounded in the composite
# keyword annotations. Careers without a data-grounded prior are reported as
# "unspecified" rather than assigned a fabricated leaning.
CAREER_GENDER_PRIOR = {
    "barber": "male",
    "butcher": "male",
    "fisherman": "male",
    "kindergarten teacher": "female",
    "mechanic": "male",
    "midwife": "female",
    "nanny": "female",
    "receptionist": "female",
    "social worker": "female",
}

DATASET_PROFILES = {
    "canonical-stage1-6900-v1": (
        "description_career_bias.json",
        "descriptions_persona_bias.json",
        "descriptions_status_bias.json",
    ),
    "legacy-5900-v1": (
        "description_career_bias.json",
        "descriptions_persona_bias.json",
        "descriptions_status_bias.json",
        "descriptions_two_axis.json",
        "descriptions_multi_axis.json",
    ),
}

DATASET_HASHES = {
    "canonical-stage1-6900-v1": {
        "description_career_bias.json": "faffa2b10ce4cdc201054fe6470270028c2a9f373d5c9ba60fd521ff45ead1fd",
        "descriptions_persona_bias.json": "ee973d75bc12118f0165a658c64d6ff77ea49561b5bbbd0383254381a067469f",
        "descriptions_status_bias.json": "ff729a1bace361d8bed732a1d27ad58bd0646eaf270eaadd69670f86ed6f88b9",
    },
    "legacy-5900-v1": {
        "description_career_bias.json": "89f035079ef38d619cbb9414b55fbefa75e441590fe77be6a49dc7c4c483503f",
        "descriptions_persona_bias.json": "5cba5fdd33fd93ca642e760b24dbb94466059028f425c9322edabe898d23fda5",
        "descriptions_status_bias.json": "d83fc2fbb6c0762de2de5f20fa91c2e96c381134fa35075ca6d140e74652c61b",
        "descriptions_two_axis.json": "92bb498e2ae283c283c4b3deefdae16abfb4b6b822e37334d4574ae92838273e",
        "descriptions_multi_axis.json": "dc338e288b2b0223a12e3cb77b841d66d3c879e4a48cd4995f25050361cdb260",
    },
}

# Single-axis files map cleanly onto one axis; composite files are multi-axis.
_SINGLE_AXIS = {
    "description_career_bias.json": "career",
    "descriptions_persona_bias.json": "persona",
    "descriptions_status_bias.json": "status",
}
_COMPOSITE = {
    "descriptions_two_axis.json": "two_axis",
    "descriptions_multi_axis.json": "multi_axis",
}


def parse_keywords(keywords):
    """Return a dict of the ``key=value`` pairs encoded in a keyword string."""
    fields = {}
    for part in (keywords or "").split(";"):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def _descriptor_lemma(source_file, record, fields):
    """Return the stereotype-bearing lemma used as the split unit."""
    if source_file == "description_career_bias.json":
        return f"career:{record['keywords'].strip().lower()}"
    if source_file == "descriptions_persona_bias.json":
        return f"persona:{record['keywords'].strip().lower()}"
    if source_file == "descriptions_status_bias.json":
        return f"status:{record['keywords'].strip().lower()}"
    # Composite files: the lemma is the ordered tuple of present axis values.
    parts = []
    if fields.get("SDO", "none") != "none":
        parts.append(f"status={fields['SDO']}")
    if fields.get("career", "none") != "none":
        parts.append(f"career={fields['career'].lower()}")
    if fields.get("persona_trait", "none") != "none":
        parts.append(f"persona={fields['persona_trait'].lower()}")
    return f"{source_file.split('.')[0]}:" + ";".join(parts)


def _stereotype_gender(source_file, record, fields):
    """Return the analysis-only stereotype gender label or ``unspecified``."""
    if source_file == "description_career_bias.json":
        return CAREER_GENDER_PRIOR.get(record["keywords"].strip().lower(), "unspecified")
    if source_file == "descriptions_status_bias.json":
        return "unspecified"
    if source_file == "descriptions_persona_bias.json":
        return "unspecified"
    # Composite: prefer the strongest annotated axis bias, else unspecified.
    for key in ("SDO_bias", "career_gender", "persona_gender"):
        value = fields.get(key, "none")
        if value in ("male", "female"):
            return value
    return "unspecified"


def dataset_hashes(root, profile):
    """Return source hashes after validating a named dataset profile."""
    root = Path(root)
    try:
        expected = DATASET_HASHES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"unknown dataset profile {profile!r}; choose one of: {choices}") from exc
    actual = {}
    for source_file, digest in expected.items():
        path = root / source_file
        if not path.is_file():
            raise FileNotFoundError(f"{profile} requires {path}")
        actual[source_file] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[source_file] != digest:
            raise ValueError(f"{path} does not match the frozen {profile} hash")
    return actual


def validate_split_manifest(manifest, root, profile):
    """Reject split manifests built from another dataset profile or snapshot."""
    hashes = dataset_hashes(root, profile)
    manifest_profile = manifest.get("dataset_profile")
    if manifest_profile is None:
        # The completed experiments predate dataset-profile metadata. This is
        # the only accepted compatibility case and is guarded by frozen hashes.
        if profile != "legacy-5900-v1" or (
            manifest.get("n_records"), manifest.get("n_lemmas")
        ) != (5900, 98):
            raise ValueError("unversioned split manifest is not the frozen legacy split")
    elif manifest_profile != profile:
        raise ValueError(
            f"split profile {manifest_profile!r} does not match dataset profile {profile!r}"
        )
    elif manifest.get("source_sha256") != hashes:
        raise ValueError("split source hashes do not match the selected dataset")
    return hashes


def load_descriptors(root, profile, *, verify=True):
    """Load and normalise one explicit, versioned descriptor profile."""
    root = Path(root)
    try:
        descriptor_files = DATASET_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_PROFILES))
        raise ValueError(f"unknown dataset profile {profile!r}; choose one of: {choices}") from exc
    if verify:
        dataset_hashes(root, profile)
    records = []
    for source_file in descriptor_files:
        path = root / source_file
        if not path.is_file():
            raise FileNotFoundError(f"{profile} requires {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        axis = _SINGLE_AXIS.get(source_file) or _COMPOSITE[source_file]
        for record in data:
            fields = parse_keywords(record.get("keywords", ""))
            records.append(
                {
                    "uid": f"{source_file.split('.')[0]}:{record['id']}",
                    "source_file": source_file,
                    "axis": axis,
                    "trait": record.get("trait", ""),
                    "description": record["description"],
                    "transcript": record["prompt_text"],
                    "lemma": _descriptor_lemma(source_file, record, fields),
                    "stereotype_gender": _stereotype_gender(source_file, record, fields),
                    "fields": fields,
                }
            )
    return records


def assign_splits(lemmas, *, seed, dev_frac=0.2, test_frac=0.4):
    """Deterministically map lemmas to train/dev/test without overlap.

    Splitting is performed at the lemma level so that no stereotype descriptor
    that trains or tunes the intervention also appears in the test evaluation.
    """
    import random

    ordered = sorted(set(lemmas))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_test = round(n * test_frac)
    n_dev = round(n * dev_frac)
    assignment = {}
    for index, lemma in enumerate(ordered):
        if index < n_test:
            assignment[lemma] = "test"
        elif index < n_test + n_dev:
            assignment[lemma] = "dev"
        else:
            assignment[lemma] = "train"
    return assignment
