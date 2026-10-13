"""Step 1: split a trial's eligibility text into atomic, checkable criteria."""

from __future__ import annotations

import hashlib
import json
import re

from .config import Settings
from .llm import StructuredLLM
from .models import AtomicCriterion, LLMDecomposition, Threshold, Trial
from .verify import locate_quote

# Bump when the prompt changes so cached decompositions are not reused.
PROMPT_VERSION = "3"

SYSTEM_PROMPT = """\
You prepare clinical trial eligibility criteria for a screening tool. Each item you produce will be \
checked, one at a time, against a patient's medical note, and a clinician will then review every \
determination. An item is only useful if it can be answered with a single look-up in the note, so \
your job is to break the criteria text into atomic criteria.

How to split
- One checkable fact per item. "Age 18 or older and ECOG 0-1" becomes two items. A lab panel \
("ANC >= 1.5, platelets >= 100, hemoglobin >= 9") becomes one item per test.
- A qualifier that adds a second requirement is an AND even without the word: "WHO performance \
status 0-1 with no deterioration over the previous 2 weeks" is two items, the status and the absence \
of deterioration. Splitting these matters because a note often settles one part and not the other, \
and an unsplit item would then be reported as wholly unknown.
- Split only where the parts are joined by AND, i.e. each part must hold on its own. Keep \
alternatives (OR) and exceptions ("unless", "except", "with the exception of") together in one item: \
splitting them would change the meaning.
- Each item's `text` must stand alone. Resolve references such as "the above" or "study drug" using \
the surrounding text, and carry any shared qualifier or time window into every item it applies to.
- Do not add, drop, strengthen or weaken any requirement.
- Keep each item pointing the way the registry wrote it. An exclusion item states the condition that \
excludes a patient, as a positive statement that is true of a patient who must be kept out: "Prior \
treatment with an EGFR-TKI" becomes "The patient has previously received an EGFR-TKI", and "ANC below \
LLN" becomes "The patient's absolute neutrophil count is below the lower limit of normal". Never \
rewrite an exclusion as the requirement a patient has to satisfy ("has not received...", "is not \
below..."). The next step judges whether each statement is true of the patient, and the program then \
treats a true exclusion as blocking enrollment, so a negated exclusion would silently reverse the \
result and rule out exactly the patients who qualify. Inclusion items likewise state what must be \
true of a patient who can enroll.
- Keep criteria that a medical note may not be able to answer (consent, willingness, investigator \
judgment). The clinician still needs to see them.
- Skip text that is not a criterion: section headings, notes to investigators, definitions.

Fields
- `type`: "inclusion" or "exclusion", taken from the section the text appears in. If there are no \
section headings, decide from the wording.
- `source_text`: copy the original span this item came from, character for character, so the \
clinician can see what was actually written. Several items split from one sentence share the same \
source_text.
- `time_window`: any time constraint as written ("within 28 days before randomization"), else null.
- `threshold`: for numeric criteria (lab values, age, dose, scores), the parameter, comparator, value \
and unit as written; else null.
- `concepts`: the diseases, drugs, lab tests and procedures the item mentions. `original` is the \
wording used; `normalized` is the standard preferred name (generic drug name, standard disease or \
test name, abbreviation expanded). If you are not confident of a standard name, set `normalized` to \
null rather than choosing the nearest match.

- `answer_source`: "screening_visit" for criteria that only the patient or the investigator can \
settle in person at screening: consent, willingness (for example to use contraception), ability to \
comply with the protocol, an estimate of life expectancy, anything left to the investigator's opinion. \
"medical_record" for everything a patient's records could document, including history, diagnoses, \
treatments, allergies, lab values and test results. The tool reports the two groups separately so \
that a clinician can tell "ask at the visit" apart from "missing from the note".

List inclusion criteria first, then exclusion criteria, each in their original order."""

_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$")


def _cache_key(trial: Trial, model: str) -> str:
    digest = hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{trial.criteria_text}".encode()).hexdigest()
    return f"{trial.nct_id}-{digest[:12]}"


def _age_threshold(age: str, comparator: str) -> Threshold:
    match = _AGE_RE.match(age)
    value, unit = (match.group(1), match.group(2)) if match else (age, None)
    return Threshold(parameter="age", comparator=comparator, value=value, unit=unit)


def structured_field_criteria(trial: Trial, decomposed: list[AtomicCriterion]) -> list[AtomicCriterion]:
    """Criteria that exist only as registry fields (age limits, sex).

    An age limit is skipped when the free-text criteria already state one in
    the same direction, to avoid showing the clinician the same check twice.
    """
    age_comparators = {
        c.threshold.comparator
        for c in decomposed
        if c.type == "inclusion" and c.threshold and c.threshold.parameter.strip().lower() == "age"
    }
    extra: list[AtomicCriterion] = []

    def add(text: str, source: str, threshold: Threshold | None = None) -> None:
        extra.append(
            AtomicCriterion(
                id="", type="inclusion", text=text, source_text=source, origin="structured_field", threshold=threshold
            )
        )

    if trial.minimum_age and not age_comparators & {">=", ">", "between"}:
        add(f"Age at least {trial.minimum_age}", f"Minimum age: {trial.minimum_age}", _age_threshold(trial.minimum_age, ">="))
    if trial.maximum_age and not age_comparators & {"<=", "<", "between"}:
        add(f"Age at most {trial.maximum_age}", f"Maximum age: {trial.maximum_age}", _age_threshold(trial.maximum_age, "<="))
    if trial.sex in ("FEMALE", "MALE"):
        add(f"Sex is {trial.sex.lower()}", f"Sex: {trial.sex}")
    return extra


def assign_ids(criteria: list[AtomicCriterion]) -> list[AtomicCriterion]:
    counters = {"inclusion": 0, "exclusion": 0}
    prefixes = {"inclusion": "INC", "exclusion": "EXC"}
    ordered = [c for c in criteria if c.type == "inclusion"] + [c for c in criteria if c.type == "exclusion"]
    for criterion in ordered:
        counters[criterion.type] += 1
        criterion.id = f"{prefixes[criterion.type]}-{counters[criterion.type]:02d}"
    return ordered


def decompose_trial(trial: Trial, settings: Settings, llm: StructuredLLM) -> list[AtomicCriterion]:
    cache_path = settings.decomposition_dir / f"{_cache_key(trial, settings.model)}.json"
    if cache_path.is_file():
        return [AtomicCriterion.model_validate(c) for c in json.loads(cache_path.read_text(encoding="utf-8"))]

    result = llm.structured(
        system=SYSTEM_PROMPT,
        user_blocks=[
            f"Trial: {trial.nct_id} - {trial.title}\n"
            f"Conditions: {', '.join(trial.conditions) or 'not listed'}\n\n"
            f"<eligibility_criteria>\n{trial.criteria_text}\n</eligibility_criteria>"
        ],
        schema=LLMDecomposition,
    )
    decomposed = [
        AtomicCriterion(
            id="",
            type=item.type,
            text=item.text.strip(),
            source_text=item.source_text.strip(),
            source_verified=locate_quote(trial.criteria_text, item.source_text) is not None,
            time_window=item.time_window,
            threshold=item.threshold,
            concepts=item.concepts,
            answer_source=item.answer_source,
        )
        for item in result.criteria
        if item.text.strip()
    ]
    criteria = assign_ids(structured_field_criteria(trial, decomposed) + decomposed)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([c.model_dump() for c in criteria], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return criteria
