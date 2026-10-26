import re

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
LETTER = re.compile(r"[^\W\d_]")


def language(text: str) -> str:
    """Rough language guess: 'ru' if most letters are Cyrillic, else 'en'."""
    letters = LETTER.findall(text)
    if not letters:
        return "unknown"
    cyrillic = sum(1 for letter in letters if CYRILLIC.match(letter))
    return "ru" if cyrillic / len(letters) > 0.5 else "en"
