"""Dependency-free statistics implementing the paper's equations 5 and 6."""

import math
import random

def binary_labels(labels):
    """Return adult binary labels and counts excluded by classifier outcome."""
    values = []
    excluded = {"child": 0, "unknown": 0, "other": 0}
    for label in labels:
        normalized = str(label).strip().lower()
        if normalized == "female":
            values.append(1)
        elif normalized == "male":
            values.append(0)
        elif normalized in excluded:
            excluded[normalized] += 1
        else:
            excluded["other"] += 1
    return values, excluded


def paper_labels(labels):
    """Encode the paper's female event over successful classifier outcomes.

    Female is 1; male and child are non-female (0). Unknown and unrecognized
    outcomes are excluded because D(y) did not produce a classifier outcome.
    Adult-only labels are returned separately for transparent secondary rates.
    """
    values = []
    adult = []
    excluded = {"unknown": 0, "other": 0}
    counts = {"female": 0, "male": 0, "child": 0}
    for label in labels:
        normalized = str(label).strip().lower()
        if normalized in counts:
            counts[normalized] += 1
            values.append(1 if normalized == "female" else 0)
            if normalized != "child":
                adult.append(1 if normalized == "female" else 0)
        elif normalized == "unknown":
            excluded["unknown"] += 1
        else:
            excluded["other"] += 1
    return values, adult, counts, excluded


def _corrected_counts(labels):
    successes = sum(labels)
    failures = len(labels) - successes
    # Boundary-only Haldane-Anscombe correction avoids infinite empirical
    # logits while leaving every interior binomial proportion unchanged.
    if successes == 0 or failures == 0:
        return successes + 0.5, failures + 0.5
    return float(successes), float(failures)


def log_odds(labels):
    """L(x)=ln(P(x)/(1-P(x))) with a binomial boundary correction."""
    if not labels:
        raise ValueError("condition has no classifier outcomes")
    successes, failures = _corrected_counts(labels)
    return math.log(successes / failures)


def interaction(groups, coefficients):
    """Compute a linear combination of condition logits.

    Equation 5 coefficients are [1, -1, -1] for joint, uni1, uni2.
    Expanded equation 6 coefficients are [1, -1, -1, -1, 1, 1, 1]
    for triple, three pairs, and three univariate conditions.
    """
    if len(groups) != len(coefficients):
        raise ValueError("groups and coefficients must have equal lengths")
    return sum(coef * log_odds(group) for group, coef in zip(groups, coefficients))


def _solve(matrix, vector):
    """Solve a small dense linear system with partial pivoting."""
    size = len(vector)
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("constrained null is numerically singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def constrained_null_probabilities(groups, coefficients):
    """Fit independent binomials subject to sum(c_i logit(p_i)) = 0."""
    if len(groups) != len(coefficients) or not groups or not any(coefficients):
        raise ValueError("a non-empty interaction contrast is required")
    adjusted = [_corrected_counts(group) for group in groups]
    totals = [success + failure for success, failure in adjusted]
    successes = [success for success, _ in adjusted]
    logits = [math.log(success / failure) for success, failure in adjusted]
    multiplier = 0.0
    for _ in range(100):
        probabilities = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value)))) for value in logits]
        gradient = [
            success - total * probability + coefficient * multiplier
            for success, total, probability, coefficient
            in zip(successes, totals, probabilities, coefficients)
        ]
        constraint = sum(coefficient * value for coefficient, value in zip(coefficients, logits))
        residual = gradient + [constraint]
        if max(abs(value) for value in residual) < 1e-10:
            return probabilities
        size = len(groups)
        jacobian = [[0.0] * (size + 1) for _ in range(size + 1)]
        for index, (total, probability, coefficient) in enumerate(zip(totals, probabilities, coefficients)):
            jacobian[index][index] = -total * probability * (1.0 - probability)
            jacobian[index][-1] = coefficient
            jacobian[-1][index] = coefficient
        step = _solve(jacobian, [-value for value in residual])
        logits = [max(-30.0, min(30.0, value + delta)) for value, delta in zip(logits, step)]
        multiplier += step[-1]
    raise ValueError("constrained null did not converge")


def permutation_test(groups, coefficients, iterations=10000, seed=0):
    """Two-sided constrained-null random-label test for an interaction.

    The paper states random-label shuffling but does not publish its algorithm.
    A pooled shuffle would test equal probabilities rather than I=0. We instead
    fit binomial probabilities under the additive-logit constraint I=0 and draw
    random binary labels at fixed condition sample sizes. The +1 Monte Carlo
    correction prevents zero p-values.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    observed = interaction(groups, coefficients)
    sizes = [len(group) for group in groups]
    if any(size == 0 for size in sizes):
        raise ValueError("every condition needs at least one valid classification")
    probabilities = constrained_null_probabilities(groups, coefficients)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        randomized = [
            [1 if rng.random() < probability else 0 for _ in range(size)]
            for probability, size in zip(probabilities, sizes)
        ]
        if abs(interaction(randomized, coefficients)) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (iterations + 1)
    if p_value < 0.01 and abs(observed) > 2.8:
        significance = "strong"
    elif p_value < 0.05 and abs(observed) > 1.0:
        significance = "moderate"
    else:
        significance = "not_significant"
    return observed, p_value, significance


def validate_interaction_spec(spec):
    errors = []
    interactions = spec.get("interactions") if isinstance(spec, dict) else None
    if not isinstance(interactions, list) or not interactions:
        return ["spec must contain a non-empty interactions list"]
    for index, item in enumerate(interactions):
        prefix = f"interactions[{index}]"
        order = item.get("order") if isinstance(item, dict) else None
        required = 3 if order == 2 else 7 if order == 3 else None
        conditions = item.get("conditions", []) if isinstance(item, dict) else []
        if required is None:
            errors.append(f"{prefix}.order must be 2 or 3")
        elif len(conditions) != required:
            errors.append(f"{prefix} requires exactly {required} complete conditions")
        names = [condition.get("name") for condition in conditions if isinstance(condition, dict)]
        if len(names) != len(set(names)):
            errors.append(f"{prefix} condition names must be unique")
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
            errors.append(f"{prefix}.name must be a non-empty string")
        for condition in conditions:
            if not isinstance(condition, dict) or not condition.get("name") or not condition.get("csv"):
                errors.append(f"{prefix} conditions require name and csv")
    return errors
