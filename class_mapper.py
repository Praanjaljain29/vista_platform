from __future__ import annotations

SYNONYM_MAP: dict[str, list[str]] = {
    "automobile": ["car"],
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "two-wheeler": ["motorcycle", "bicycle"],
    "motorbike": ["motorcycle"],
    "bike": ["bicycle", "motorcycle"],
    "pedestrian": ["person"],
    "human": ["person"],
    "individual": ["person"],
    "man": ["person"],
    "woman": ["person"],
    "child": ["person"],
    "kid": ["person"],
    "cyclist": ["person", "bicycle"],
    "driver": ["person"],
    "passenger": ["person"],
}

_UNVERIFIABLE: set[str] = {
    # colors
    "red", "blue", "green", "black", "white", "yellow",
    "orange", "purple", "pink", "brown", "grey", "gray",
    # clothing
    "shirt", "jacket", "dress", "hat", "coat", "pants", "shoes",
    # size modifiers
    "small", "large", "big", "tall", "short",
}


class SemanticClassMapper:
    """Maps natural-language concepts to YOLO detector class names."""

    @staticmethod
    def resolve(concept: str | None, available_classes: set[str]) -> list[str]:
        """Return the list of available YOLO classes that match *concept*.

        1. Returns [] if concept is None or empty.
        2. Normalises concept to lowercase/stripped.
        3. If the normalised concept is directly in available_classes, returns it.
        4. Otherwise looks up SYNONYM_MAP and filters to available_classes.
        5. Returns the filtered list (may be empty).
        """
        if not concept:
            return []
        normalized = concept.strip().lower()
        # Normalise available_classes to lowercase for comparison
        lower_available = {c.lower() for c in available_classes}
        if normalized in lower_available:
            return [normalized]
        candidates = SYNONYM_MAP.get(normalized, [])
        return [c for c in candidates if c.lower() in lower_available]

    @staticmethod
    def describe_unsupported(attributes: list[str]) -> list[str]:
        """Return the subset of *attributes* that cannot be visually verified
        by the detector (colors, clothing references, size modifiers)."""
        return [attr for attr in attributes if attr.strip().lower() in _UNVERIFIABLE]
