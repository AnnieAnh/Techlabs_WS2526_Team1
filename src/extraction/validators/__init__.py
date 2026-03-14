"""Post-extraction validators for the job extraction pipeline."""

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Optional


@dataclass
class ValidationFlag:
    """A single validation finding for a row.

    Attributes:
        row_id: The row this flag applies to.
        field: The specific field that triggered the flag.
        rule: The rule name (e.g. 'min_greater_than_max', 'skill_not_in_description').
        severity: 'error', 'warning', or 'info'.
        message: Human-readable description of the issue.
        context: Optional structured data for machine-readable details (e.g. {"skill": "Python"}).
    """

    row_id: str
    field: str
    rule: str
    severity: str
    message: str
    context: Optional[dict] = dc_field(default=None)
