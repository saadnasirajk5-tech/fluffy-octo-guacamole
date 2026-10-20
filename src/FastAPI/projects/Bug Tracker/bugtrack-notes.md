
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

Bro, say less. We're doing this properly — no copy-paste tutorial hell, we build understanding first, code second. And listen: FastAPI isn't a random pick for you. As a future security auditor, you *will* write PoC servers, exploit tooling, and read APIs to find their attack surface. Backend fluency is a weapon for your path.

---

## 1. The mental model (get this right first)

When any app talks to a server, three things happen:

1. A **client** (browser, mobile app, script) sends an **HTTP request**: a method (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`) + a path (`/findings`) + optional body (data).
2. A **server** receives it, runs your logic, and returns an **HTTP response**: a status code + data (usually JSON).
3. A **framework** (FastAPI) is the glue: it matches the request to your function, validates the input, and formats the output.

FastAPI itself doesn't run anything — **uvicorn** is the actual server that listens on a port. FastAPI is the brain, uvicorn is the ears and mouth. That's why the run command is `uvicorn main:app` — "in the file `main.py`, find the variable `app`, and serve it."

## 2. The five ideas that are 80% of FastAPI

**① The app instance + routes.** A route = "when someone hits this path, run this function."

```python
app = FastAPI()

@app.get("/hello")
def hello():
    return {"message": "yo"}
```

The `@app.get` decorator *registers* the function; it doesn't call it. FastAPI auto-converts the returned dict to JSON — that's the framework doing the boring work for you.

**② Path parameters + type hints.** Put a variable in the path, and FastAPI extracts it *and validates its type*:

```python
@app.get("/findings/{finding_id}")
def get_finding(finding_id: int):
    ...
```

If someone hits `/findings/abc`, they get an automatic **422** error before your code even runs. Never trust input — this one line just eliminated a whole class of bugs.

**③ Request bodies with Pydantic.** You declare what valid data looks like, FastAPI enforces it at the border:

```python
class FindingCreate(BaseModel):
    title: str = Field(min_length=3, max_length=100)
    severity: Severity  # an Enum whitelist: low/medium/high/critical
```

Send `{"severity": "nuclear"}` or a 10,000-character title → rejected with a clear error. **This is the security mindset**: validation at the edge = smaller attack surface. Whitelist what's allowed, don't blacklist what's not.

**④ Status codes + errors.** HTTP has a contract, and professionals respect it:
- `200` OK (GET, PATCH)
- `201` Created (POST — you made something new)
- `204` No Content (DELETE — success, nothing to say)
- `404` Not Found — you raise `HTTPException(status_code=404, detail="...")`
- `422` Unprocessable — FastAPI does this automatically when validation fails

**⑤ Free docs.** Go to `/docs` on any running FastAPI app and you get an interactive Swagger UI generated from your code. This is why companies adopted it so fast.

## 3. Your app — built

I built the **Findings API** in the canvas: a mini bug-tracker where you create, list, filter, update, and delete security findings. It's one file on purpose — in Lesson 1 I want you to see the framework naked, not buried in project structure. Every section is commented with the *why*. Real trade-off to understand: storage is in-memory (a dict), so data dies on restart — that's deliberate for Lesson 1; a real database is Lesson 2.

## 4. Run it, then deploy it

**Locally:**
```bash
pip install fastapi "uvicorn[standard]"
uvicorn main:app --reload
```
Then open `http://127.0.0.1:8000/docs` and click around. The `--reload` flag restarts the server on every save — dev mode only.

**Deploy (free, ~10 minutes):**
1. Put the app in a GitHub repo as `main.py`, plus a `requirements.txt` containing:
   ```
   fastapi
   uvicorn[standard]
   ```
2. On [Render](https://render.com), create a **New → Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT` — `$PORT` must stay a variable because Render picks the port itself; hardcode one and the deploy fails ([Render docs](https://render.com/docs/deploy-fastapi), [Seekvana guide](https://seekvana.com/library/building-with-ai/deploy-to-render)).
4. Deploy. You get a live `https://yourapp.onrender.com` URL with `/docs` working.

Two honest caveats about the free tier: the app **sleeps after 15 minutes of inactivity**, so the first request after idle takes 30–60 seconds — that's not a broken deploy ([Kuberns](https://kuberns.com/blogs/deploy-fastapi-on-render/)) — and each restart wipes your in-memory data. Both are exactly why Lesson 2 is databases.

## 5. Your homework (this is where you escape tutorial hell)

Don't copy the code from the canvas. **Type it yourself**, deploy it, then deliberately break things and read the errors:

- POST a finding with `severity: "catastrophic"` → watch the 422 tell you exactly what's allowed
- `GET /findings/999` → watch your 404
- `PATCH` with `state: "deleted"` → watch the regex validation reject it
- Delete the `response_model=Finding` line and send bad data in — see what the difference is

If you can explain *why* each of those behaved the way it did, you've understood more than most people who finished three courses. Report back with your live URL and we go to Lesson 2: real database with SQLite, then authentication (you'll love that one — it's where most real-world vulns live).

I also saved your FastAPI learning progress and teaching preferences, so next session we pick up right where we left off. 📚 [Career Direction](knowledge://knowledge/career-direction/KNOWLEDGE.md)