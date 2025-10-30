"""Parse digit labels from natural language prompts."""

import re

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
}


def parse_text_to_digit(text: str) -> int:
    """Extract digit (0-9) from text prompt.

    Supports: "Print number 3", "Generate five", "Show digit 7", "3", "three"
    """
    text_lower = text.lower().strip()

    for word, digit in NUMBER_WORDS.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            return digit

    digit_match = re.search(r'\b([0-9])\b', text_lower)
    if digit_match:
        return int(digit_match.group(1))

    raise ValueError(
        f"Could not extract digit from: '{text}'. "
        f"Use patterns like 'Print number 3', 'five', or just '7'."
    )


def digit_to_onehot(digit: int, num_classes: int = 10) -> list:
    """Convert digit to one-hot list."""
    if not 0 <= digit < num_classes:
        raise ValueError(f"Digit must be in [0, {num_classes-1}], got {digit}")

    onehot = [0] * num_classes
    onehot[digit] = 1
    return onehot


def parse_text_to_onehot(text: str, num_classes: int = 10) -> list:
    """Parse text and return one-hot encoded label."""
    digit = parse_text_to_digit(text)
    return digit_to_onehot(digit, num_classes)
