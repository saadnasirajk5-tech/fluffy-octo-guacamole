"""
Findings API — a tiny bug-tracker backend.
Lesson 1 of the FastAPI course: one file, no database, so the framework
itself stays visible. Storage is in-memory, so data resets on restart —
that trade-off is deliberate and explained in the lesson.
Run locally:   uvicorn main:app --reload
Deploy start:  uvicorn main:app --host 0.0.0.0 --port $PORT
"""
from enum import Enum
from itertools import count

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
app = FastAPI(
    title="Findings API",
    description="A tiny API that stores security findings — like a mini bug tracker.",
    version="1.0.0",
)
# ---------------------------------------------------------------
# 1) Validation models (Pydantic)
#    The single source of truth for what data is allowed in.
# ---------------------------------------------------------------
class Severity(str, Enum):
    """Whitelist of allowed severity values. Anything else -> 422 rejected."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FindingCreate(BaseModel):
    """The contract a client must satisfy to create a finding."""
    title: str = Field(min_length=3, max_length=100)
    severity: Severity
    description: str = Field(default="", max_length=2000)

class FindingUpdate(BaseModel):
    """Partial update: only the state can change, and only to open|fixed."""
    state: str = Field(pattern="^(open|fixed)$")

class Finding(FindingCreate):
    """What we return: the client's data plus an id and a state."""
    id: int
    state: str = "open"

# ---------------------------------------------------------------
# 2) Storage
#    In-memory dict: fast and simple, but wiped on every restart.
#    Lesson 2 replaces this with a real database.
# ---------------------------------------------------------------
findings_db: dict[int, Finding] = {}
_id_factory = count(start=1)

def _next_id() -> int:
    return next(_id_factory)

def _get_or_404(finding_id: int) -> Finding:
    if finding_id not in findings_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Finding {finding_id} not found",
        )
    return findings_db[finding_id]

# ---------------------------------------------------------------
# 3) Endpoints
# ---------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Findings API",
        "docs": "/docs",
        "example": "POST /findings with {'title': 'Reentrancy in deposit()', 'severity': 'critical'}",
    }


@app.post(
    "/findings",
    response_model=Finding,
    status_code=status.HTTP_201_CREATED,
)
def create_finding(payload: FindingCreate):
    """Create a finding. 201 because a new resource was created."""
    finding = Finding(id=_next_id(), **payload.model_dump())
    findings_db[finding.id] = finding
    return finding


@app.get("/findings", response_model=list[Finding])
def list_findings(severity: Severity | None = None, state: str | None = None):
    """List findings. Optional query params filter, e.g. /findings?severity=critical."""
    results = list(findings_db.values())
    if severity is not None:
        results = [f for f in results if f.severity == severity]
    if state is not None:
        results = [f for f in results if f.state == state]
    return results


@app.get("/findings/{finding_id}", response_model=Finding)
def get_finding(finding_id: int):
    """Fetch one finding. finding_id is typed int, so /findings/abc -> 422."""
    return _get_or_404(finding_id)


@app.patch("/findings/{finding_id}", response_model=Finding)
def update_finding(finding_id: int, patch: FindingUpdate):
    """Change the state of a finding. Only 'open' or 'fixed' pass validation."""
    finding = _get_or_404(finding_id)
    finding.state = patch.state
    return finding


@app.delete("/findings/{finding_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_finding(finding_id: int):
    """Delete a finding. 204 = success with no response body."""
    _get_or_404(finding_id)
    del findings_db[finding_id]