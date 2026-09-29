"""Whether a spoken name is the name on the account.

The one identity answer that arrives as a word, not a number — so a speech
model has to pick a spelling for it. Both cases below were MEASURED in
simulation, not invented: a caller who said her name plainly was refused
because the transcript read "Sorenson" against an account reading "Sorensen",
and another was refused three times over "Oconquo" and "O'Conquil" while her
account number and postal code matched exactly every attempt.

Exact matching tests whether a speech model spells the way the account does —
a different question from whether the caller knows their own name, and one
that goes worst for names the model saw least. So these pin sound, not
spelling, and pin that the ten seeded accounts stay distinct from each other.
"""

import pytest

from rory_tools.impl import _name_matches

SEEDED = [
    "Alice Okonkwo", "Ben Castellano", "Carmen Reyes", "Dmitri Volkov",
    "Elena Ferrara", "Femi Adeyemi", "Grace Lindqvist", "Hassan Malik",
    "Ingrid Sorensen", "Jonah Pratt",
]


@pytest.mark.parametrize("heard, on_file", [
    ("Ingrid Sorensen", "Ingrid Sorensen"),   # transcribed cleanly
    ("Ingrid Sorenson", "Ingrid Sorensen"),   # measured: -son for -sen
    ("Alice Oconquo", "Alice Okonkwo"),       # measured
    ("Alice O'Conquil", "Alice Okonkwo"),     # measured, third attempt
    ("alice okonkwo", "Alice Okonkwo"),       # case
    ("  Carmen   Reyes ", "Carmen Reyes"),    # spacing
    ("Dmitry Volkov", "Dmitri Volkov"),       # -y for -i
])
def test_a_caller_who_said_their_own_name_is_recognised(heard, on_file):
    assert _name_matches(on_file, heard) is True


@pytest.mark.parametrize("heard, on_file", [
    ("Ellis Oconquo", "Alice Okonkwo"),       # measured: wrong FIRST name
    ("Carmen Reyes", "Ingrid Sorensen"),
    ("Ben Castellano", "Alice Okonkwo"),
    ("Alice Castellano", "Alice Okonkwo"),    # right first, wrong surname
    ("Alice", "Alice Okonkwo"),               # a first name is not a name
    ("Alice Okonkwo Reyes", "Alice Okonkwo"), # more parts than the account has
    ("", "Alice Okonkwo"),
    (None, "Alice Okonkwo"),
])
def test_a_name_that_is_not_the_one_on_file_is_refused(heard, on_file):
    assert _name_matches(on_file, heard) is False


def test_no_two_seeded_accounts_share_a_name():
    """The check is only as safe as the population it runs against.

    Phonetic matching trades exactness for tolerance, so the thing to pin is
    that it never lets one real customer answer as another.
    """
    for a in SEEDED:
        for b in SEEDED:
            if a != b:
                assert _name_matches(a, b) is False, f"{a} matched {b}"
