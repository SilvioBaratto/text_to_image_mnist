"""
Text parser for extracting digit labels from natural language commands.
Supports patterns like "Print number 3", "Generate five", "Show digit 7", etc.
"""

import re
from typing import Optional


# Mapping from number words to digit values
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9
}


def parse_text_to_digit(text: str) -> int:
    """
    Extract digit (0-9) from natural language text command.

    Supported patterns:
    - "Print number 3" / "Print 3"
    - "Generate number three" / "Generate three"
    - "Show digit 5" / "Show 5"
    - Just "3" or "three"

    Args:
        text: Natural language text prompt

    Returns:
        digit: Integer digit value (0-9)

    Raises:
        ValueError: If no valid digit is found in the text
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower().strip()

    # First, try to find number words (more specific than digit patterns)
    for word, digit in NUMBER_WORDS.items():
        # Use word boundaries to avoid matching substrings
        if re.search(r'\b' + word + r'\b', text_lower):
            return digit

    # Then, try to find digit characters (0-9)
    digit_match = re.search(r'\b([0-9])\b', text_lower)
    if digit_match:
        digit = int(digit_match.group(1))
        return digit

    # If no valid digit found, raise error with helpful message
    raise ValueError(
        f"Could not extract digit from text: '{text}'. "
        f"Please use patterns like 'Print number 3', 'Generate five', 'Show digit 7', "
        f"or simply '3' or 'three'. Valid digits are 0-9."
    )


def digit_to_onehot(digit: int, num_classes: int = 10) -> list:
    """
    Convert digit to one-hot encoded list.

    Args:
        digit: Integer digit value (0-9)
        num_classes: Number of classes (default: 10 for MNIST)

    Returns:
        onehot: One-hot encoded list of length num_classes

    Raises:
        ValueError: If digit is out of range [0, num_classes-1]
    """
    if not 0 <= digit < num_classes:
        raise ValueError(f"Digit must be in range [0, {num_classes-1}], got {digit}")

    onehot = [0] * num_classes
    onehot[digit] = 1
    return onehot


def parse_text_to_onehot(text: str, num_classes: int = 10) -> list:
    """
    Parse text prompt and convert to one-hot encoded label.

    Args:
        text: Natural language text prompt
        num_classes: Number of classes (default: 10 for MNIST)

    Returns:
        onehot: One-hot encoded list of length num_classes

    Raises:
        ValueError: If no valid digit is found in text or digit is out of range
    """
    digit = parse_text_to_digit(text)
    onehot = digit_to_onehot(digit, num_classes)
    return onehot
