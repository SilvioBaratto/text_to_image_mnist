"""Generate diverse natural language prompts for MNIST digit generation."""

import random

# Digit name mappings
DIGIT_NAMES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

# Template categories for diverse prompts
PROMPT_TEMPLATES = [
    # Direct requests
    "Draw the digit {digit}",
    "Generate number {digit}",
    "Create the digit {digit}",
    "Show me the number {digit}",
    "Make a {digit}",
    "Print number {digit}",

    # Descriptive requests
    "I want to see a {digit_name}",
    "I want to draw a {digit_name}",
    "Can you show me a {digit_name}",
    "Can you draw a {digit_name}",
    "Please generate a {digit_name}",
    "Please create the digit {digit_name}",

    # Question forms
    "What does {digit} look like?",
    "What does the digit {digit_name} look like?",
    "How do you write {digit}?",
    "How do you draw a {digit_name}?",

    # Polite requests
    "Could you draw a {digit} for me?",
    "Would you show me the number {digit_name}?",
    "Can I see what {digit} looks like?",

    # Specific descriptions
    "Show me a handwritten {digit}",
    "Generate a handwritten {digit_name}",
    "Draw a {digit_name} digit",
    "Create the number {digit_name}",

    # Simple forms
    "{digit}",
    "{digit_name}",
    "the digit {digit}",
    "the number {digit_name}",
]


def generate_prompt_for_digit(digit: int) -> str:
    """Generate a random natural language prompt for a digit.

    Args:
        digit: Integer from 0-9

    Returns:
        str: Natural language prompt (e.g., "I want to draw a zero")
    """
    if not 0 <= digit <= 9:
        raise ValueError(f"Digit must be 0-9, got {digit}")

    template = random.choice(PROMPT_TEMPLATES)
    digit_name = DIGIT_NAMES[digit]

    # Fill in template with digit or digit_name
    prompt = template.format(digit=digit, digit_name=digit_name)

    return prompt


def generate_prompts_batch(digits: list[int]) -> list[str]:
    """Generate prompts for a batch of digits.

    Args:
        digits: List of digit integers

    Returns:
        list[str]: List of natural language prompts
    """
    return [generate_prompt_for_digit(d) for d in digits]


# For debugging/testing
if __name__ == "__main__":
    print("Sample prompts for each digit:\n")
    for digit in range(10):
        prompts = [generate_prompt_for_digit(digit) for _ in range(3)]
        print(f"Digit {digit}:")
        for p in prompts:
            print(f"  - {p}")
        print()
