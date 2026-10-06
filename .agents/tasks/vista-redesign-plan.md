# Implementation Plan — VISTA Platform Redesign

> Codebase: `c:\Users\praan\Desktop\vista_platform`
> All paths below are relative to that root unless stated otherwise.

---

## What this plan covers

Ten-part redesign of the VISTA Flask app:

1. Strip the main `/` page down to: title, upload, query, answer, matched frame
2. Add `/debug` route + `templates/debug.html` with all current technical detail
3. Upgrade `query_parser.py` to a richer nested `SearchIntent` schema (Parts 3 + 4)
4. Create `class_mapper.py` with `SemanticClassMapper` (Part 5)
5. Update `search_service.py` to consume the new intent + the mapper (Part 6)
6. Update `response_service.py` for new schema + honest unsupported-attribute handling (Part 7)
7. Update `app.py`: `_last_debug_result`, `/debug` route, pass `available_classes`, slim `render_home()` (Part 8)
8. Update the Llama parser prompt to match the new JSON schema (Part 9)
9. Syntax + import check (Part 10)

---

## Dependency order

```
Step 1  (class_mapper.py — new, standalone)
Step 2  (query_parser.py — new SearchIntent schema; no external deps except llama_service)
Step 3  (response_service.py — depends on new query_parser SearchIntent)
Step 4  (search_service.py — depends on new SearchIntent + new class_mapper)
Step 5  (app.py — depends on updated search_service, query_parser, response_service)
Step 6  (templates/index.html — main UI rewrite, depends on slimmed render_home() in Step 5)
Step 7  (templates/debug.html — new file, depends on _last_debug_result from Step 5)
Step 8  (syntax + import verification — depends on all prior steps)
```

---

## Steps

- [ ] 1. **Create `class_mapper.py`** — standalone semantic mapping module, no deps on other project files.

  Create a new file `class_mapper.py` at the project root.

  Include:

  ```python
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
  ```

  `SemanticClassMapper` class with two static methods:

  - `resolve(concept: str | None, available_classes: set[str]) -> list[str]`
    1. If concept is None/empty, return `[]`.
    2. Normalize: `concept.strip().lower()`.
    3. If normalized concept is in `available_classes` (lowercased comparison), return `[concept_normalized]`.
    4. Look up `SYNONYM_MAP.get(concept_normalized, [])` and keep only those entries that are in `available_classes`.
    5. Return the filtered list (may be empty).

  - `describe_unsupported(attributes: list[str]) -> list[str]`
    Return items from `attributes` that fall in a hardcoded set of visually unverifiable descriptors: colors (`red`, `blue`, `green`, `black`, `white`, `yellow`, `orange`, `purple`, `pink`, `brown`, `grey`, `gray`), clothing references (`shirt`, `jacket`, `dress`, `hat`, `coat`, `pants`, `shoes`), and size modifiers (`small`, `large`, `big`, `tall`, `short`). Return the subset of `attributes` that match any of those terms (case-insensitive).

  **Files:** `class_mapper.py` (new)

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from class_mapper import SemanticClassMapper; r = SemanticClassMapper.resolve('automobile', {'car','person','truck'}); assert r == ['car'], r; print('class_mapper ok')"`

---

- [ ] 2. **Upgrade `query_parser.py`** — replace flat `SearchIntent` with the nested schema; update `build_parser_prompt()`, `_fallback_intent()`, `_validate_intent()`, and `_intent_to_dict()`.

  ### 2a — Replace the `SearchIntent` dataclass

  Remove the existing `SearchIntent` dataclass. Add three new dataclasses and a new `SearchIntent`:

  ```python
  @dataclass
  class TargetEntity:
      type: str | None = None
      attributes: list[str] = field(default_factory=list)
      description: str = ""

  @dataclass
  class RelatedEntity:
      type: str | None = None
      attributes: list[str] = field(default_factory=list)

  @dataclass
  class Relation:
      type: str | None = None

  @dataclass
  class SearchIntent:
      target: TargetEntity = field(default_factory=TargetEntity)
      related_entities: list[RelatedEntity] = field(default_factory=list)
      relations: list[Relation] = field(default_factory=list)
      action: str | None = None
      temporal_constraint: str | None = None
      original_query: str = ""
      unsupported_attributes: list[str] = field(default_factory=list)
  ```

  Remove the `slots=True` argument from all four dataclasses — the nested default factories are incompatible with slots in Python < 3.10 and the codebase doesn't rely on slots for these types.

  ### 2b — Update `_fallback_intent()`

  Replace references to old fields. The fallback now:
  1. Finds a target class string with `_find_supported_object()` → sets `intent.target.type`.
  2. Finds a related class → creates a `RelatedEntity(type=related)` appended to `intent.related_entities`.
  3. Finds a relation keyword → creates `Relation(type=relation)` appended to `intent.relations`.
  4. Finds action keyword → sets `intent.action`.
  5. Collects color/size attribute words (existing logic) → sets `intent.target.attributes` and `intent.unsupported_attributes` (keep the same attribute words for both for fallback).
  6. Sets `intent.original_query = query`.
  7. Returns the populated `SearchIntent`.

  ### 2c — Update `_validate_intent()`

  Parse the JSON dict into the new nested schema:
  - `data["target"]` (dict) → `TargetEntity(type=..., attributes=..., description=...)`.
  - `data["related_entities"]` (list of dicts) → `list[RelatedEntity]`.
  - `data["relations"]` (list of dicts) → `list[Relation]`.
  - `data["action"]` → `intent.action`.
  - `data["temporal_constraint"]` → `intent.temporal_constraint`.
  - `data["unsupported_attributes"]` → `intent.unsupported_attributes` (coerce via `_coerce_list`).
  - `data["original_query"]` → `intent.original_query` (fall back to `raw_query` argument).
  - If `intent.target.type` is None after parsing, call `_fallback_intent(raw_query)` and return it.

  ### 2d — Update `_intent_to_dict()`

  Return a dict matching the new schema shape (nested target, related_entities list, relations list, etc.) for use in debug output.

  ### 2e — Update `build_parser_prompt()`

  Replace the old flat schema example with the new nested one. See Part 9 details below in step 2f.

  ### 2f — New `build_parser_prompt()` content (Part 9)

  ```python
  def build_parser_prompt(query: str) -> str:
      schema = {
          "target": {"type": "person", "attributes": [], "description": ""},
          "related_entities": [{"type": "car", "attributes": []}],
          "relations": [{"type": "near"}],
          "action": None,
          "temporal_constraint": None,
          "unsupported_attributes": [],
          "original_query": query,
      }
      instructions = (
          "Convert the user's natural-language VISTA query into JSON only.\n"
          "Return a single JSON object and nothing else.\n"
          "Do not invent detections or evidence.\n"
          "Use null when a field is unknown.\n\n"
          "YOLO class vocabulary: person, car, truck, bus, bicycle, motorcycle, "
          "chair, table, cup, bottle, phone, laptop, tv, monitor, dog, cat, "
          "backpack, bag.\n\n"
          "Rules:\n"
          "- Map synonyms to YOLO class names: "
          "automobile→car, vehicle→car/truck/bus/motorcycle/bicycle, "
          "two-wheeler→motorcycle/bicycle, pedestrian/human→person.\n"
          "- If a concept maps to multiple classes, pick the most likely one for target.type.\n"
          "- Put unverifiable visual attributes (color, clothing, size, activity) "
          "in unsupported_attributes.\n"
          "- Return ONLY valid JSON matching the schema below.\n\n"
          f"Schema example: {json.dumps(schema)}\n\n"
          f"User query: {query}"
      )
      return instructions
  ```

  Remove the now-unused imports: `SUPPORTED_OBJECTS`, `RELATION_KEYWORDS`, `ACTION_KEYWORDS` are still needed by `_fallback_intent()` so keep them. Keep `ParseResult` unchanged.

  **Files:** `query_parser.py`

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from query_parser import parse_query, SearchIntent, TargetEntity; r = parse_query('Find the person near the car'); print('query_parser ok', r.intent.target.type)"`

---

- [ ] 3. **Update `response_service.py`** — use new `SearchIntent` fields; honest unsupported-attribute handling; clean answer (no track IDs in mock).

  ### 3a — Update `_build_response_prompt()`

  Access new fields from `parse_result.intent`:
  - `intent.target.type` (was `intent.target_object`)
  - `intent.target.attributes`
  - `intent.unsupported_attributes`
  - `intent.related_entities[0].type` if the list is non-empty (was `intent.related_object`)
  - `intent.relations[0].type` if non-empty (was `intent.relation`)
  - `intent.action`
  - `intent.temporal_constraint` (was `intent.time_window_seconds`)
  - `intent.original_query` (was `intent.raw_query`)

  Build the prompt payload dict:
  ```python
  payload = {
      "query": query,
      "intent": {
          "target": intent.target.type,
          "related_entity": intent.related_entities[0].type if intent.related_entities else None,
          "relation": intent.relations[0].type if intent.relations else None,
          "action": intent.action,
          "temporal_constraint": intent.temporal_constraint,
          "unsupported_attributes": intent.unsupported_attributes,
      },
      "search_result": evidence,
  }
  ```

  ### 3b — Update the system prompt in `generate_response()`

  Replace the existing system prompt string with:
  ```
  "You are VISTA, a concise video-search assistant. "
  "Use ONLY the provided structured evidence. "
  "Do not invent track IDs, timestamps, frames, confidence scores, or bounding boxes. "
  "Refer to objects by their class name and time only (e.g. 'around 18 seconds'). "
  "Do NOT show track IDs or confidence numbers in the answer. "
  "If unsupported_attributes is non-empty, acknowledge those attributes "
  "could not be visually verified. "
  "Give a short, human-friendly answer."
  ```

  ### 3c — Update `_mock_response()`

  Update to use `evidence.object_class` (unchanged) and `evidence.first_timestamp` but remove track ID from the mock text:
  ```python
  text = (
      f"[MOCK] Yes, I found {evidence.object_class or 'the target object'} "
      f"around {timestamp:.1f} seconds into the video."
  )
  ```
  (No Track ID / frame number in the mock answer.)

  **Files:** `response_service.py`

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from response_service import generate_response; print('response_service ok')"`

---

- [ ] 4. **Update `search_service.py`** — consume new `SearchIntent` + use `SemanticClassMapper`; accept `available_classes`.

  ### 4a — Update `VisualSearchEngine.__init__()`

  New signature:
  ```python
  def __init__(
      self,
      intent: SearchIntent,
      job_id: str,
      result_dir: str | Path,
      available_classes: set[str] | None = None,
  ) -> None:
  ```

  Store `self.available_classes = available_classes or set()`.

  Replace old flat field extraction:
  ```python
  # OLD:
  self.target_object = _normalize(intent.target_object)
  self.related_object = _normalize(intent.related_object)
  self.relation = _normalize(intent.relation)

  # NEW:
  from class_mapper import SemanticClassMapper
  self.target_classes: list[str] = SemanticClassMapper.resolve(
      intent.target.type, self.available_classes
  )
  # Fallback: if mapper returns empty and target.type is set, use it directly normalized
  if not self.target_classes and intent.target.type:
      self.target_classes = [_normalize(intent.target.type)]

  related_type = intent.related_entities[0].type if intent.related_entities else None
  self.related_classes: list[str] = SemanticClassMapper.resolve(
      related_type, self.available_classes
  )
  if not self.related_classes and related_type:
      self.related_classes = [_normalize(related_type)]

  self.relation = _normalize(intent.relations[0].type if intent.relations else None)
  self.unsupported_attributes: list[str] = intent.unsupported_attributes
  ```

  Keep `self.target_object` as `self.target_classes[0] if self.target_classes else ""` for backward compatibility in `finalize()` evidence object — or update `finalize()` to use `self.target_classes[0]` directly. Choose: update `finalize()` directly (cleaner).

  ### 4b — Update `_is_target()`

  ```python
  def _is_target(self, record: DetectionRecord) -> bool:
      if not self.target_classes:
          return False
      normalized_class = _normalize(record.object_class)
      return any(tc in normalized_class or normalized_class in tc for tc in self.target_classes)
  ```

  ### 4c — Update `_is_related()`

  ```python
  def _is_related(self, record: DetectionRecord) -> bool:
      if not self.related_classes:
          return False
      normalized_class = _normalize(record.object_class)
      return any(rc in normalized_class or normalized_class in rc for rc in self.related_classes)
  ```

  ### 4d — Update `finalize()` evidence construction

  Where the old code used `self.target_object` as a string, replace with `self.target_classes[0] if self.target_classes else None`.
  Where the old code used `self.related_object`, replace with `self.related_classes[0] if self.related_classes else None`.

  Add `notes` population: if `self.unsupported_attributes`, append a note like:
  ```python
  notes = []
  if self.unsupported_attributes:
      notes.append(
          f"Note: the following attributes could not be visually verified: "
          + ", ".join(self.unsupported_attributes)
      )
  ```
  Pass `notes=notes` in the returned `SearchResult`.

  **Files:** `search_service.py`

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from search_service import VisualSearchEngine; print('search_service ok')"`

---

- [ ] 5. **Update `app.py`** — wire everything together: `_last_debug_result`, updated `VisualSearchEngine` call, slim `render_home()`, new `/debug` route.

  ### 5a — Add module-level debug store

  At module level, after imports, add:
  ```python
  _last_debug_result: dict | None = None
  ```

  ### 5b — Update `process_video()`

  1. Update the `VisualSearchEngine` instantiation to pass `available_classes`:
     ```python
     available_classes = set(model.names.values())
     ```
     Change:
     ```python
     search_engine = VisualSearchEngine(intent, job_id, RESULT_DIR)
     ```
     to:
     ```python
     search_engine = VisualSearchEngine(intent, job_id, RESULT_DIR, available_classes=available_classes)
     ```
     Place the `available_classes` variable before the conditional that constructs `search_engine`.

  2. Update the condition that creates `search_engine`. The old guard checks `intent.target_object or intent.related_object or intent.action`. Replace with:
     ```python
     if parse_result.source != "missing" and (
         intent.target.type
         or intent.related_entities
         or intent.action
     ):
     ```

  3. The return dict from `process_video()` does NOT need to change (all the debug data is already there). However, update the `"intent"` key to use the new `_intent_to_dict(intent)` from `query_parser`:
     ```python
     from query_parser import _intent_to_dict  # add this import at the top of app.py
     ...
     "intent": _intent_to_dict(intent),
     ```

  ### 5c — Update `/process` route

  After `result = process_video(...)`, store it in `_last_debug_result`:
  ```python
  global _last_debug_result
  _last_debug_result = result
  ```

  Change the `render_home()` call to pass only the slim fields (not the full result dict):
  ```python
  return render_home(
      answer_text=result.get("assistant_message", ""),
      answer_ok=result.get("assistant_ok", False),
      matched_image=result["images"][0] if result.get("images") else None,
      query=query,
      error=None,
  )
  ```

  Also update the `except` branch to pass `answer_text=None, answer_ok=False, matched_image=None`.

  ### 5d — Update `render_home()`

  Change the signature and template call:
  ```python
  def render_home(
      answer_text: str | None,
      answer_ok: bool,
      matched_image: dict | None,
      query: str,
      error: str | None,
  ):
      return render_template(
          "index.html",
          answer_text=answer_text,
          answer_ok=answer_ok,
          matched_image=matched_image,
          query=query,
          error=error,
      )
  ```

  Update the GET `/` route to pass the new parameters (all None/empty/False).

  Also remove the `runtime_status = model_status()` call from `render_home()` — it is no longer needed on the main page (move it to the debug page only, computed inside the `/debug` route).

  ### 5e — Add `/debug` route

  ```python
  @app.route("/debug", methods=["GET"])
  def debug_page():
      from llama_service import model_status as _model_status
      return render_template(
          "debug.html",
          result=_last_debug_result,
          runtime_status=_model_status(),
      )
  ```

  ### 5f — Clean up imports in `app.py`

  Remove the now-unused import of `model_status` at the top (it was only used in `render_home()`). Import it locally in `debug_page()` instead, or keep it at the top — either is fine; just make sure no `NameError` at load time.

  **Files:** `app.py`

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from app import app; print('app imports ok')"`

---

- [ ] 6. **Rewrite `templates/index.html`** — minimal user-facing page; no debug/technical detail.

  Keep the existing dark theme CSS variables (--bg, --bg-2, --panel, --accent, --accent-2, --danger, --success, --shadow, etc.) and base body/page styles.

  Remove from CSS: `.status-card`, `.kv`, `.evidence-grid`, `.results-grid`, `.results-cards`, `.image-meta`, `.footer-links`, `pre.json`, `.results-cards`.

  ### Layout

  ```
  <div class="page">
    <!-- HEADER -->
    <header class="brand">
      <span class="eyebrow">Visual Intelligence Search & Tracking Architecture</span>
      <h1>VISTA</h1>
      <p class="subtitle">Upload a video and ask a question in plain language.</p>
    </header>

    <!-- FORM -->
    <section class="form-card">
      [video upload field]
      [query textarea]
      [Search button]
      [error banner — only if error]
    </section>

    <!-- ANSWER — only rendered when answer_text is set -->
    {% if answer_text %}
    <section class="answer-card">
      <div class="answer-label">VISTA says:</div>
      <p class="answer-text">{{ answer_text }}</p>
    </section>
    {% endif %}

    <!-- MATCHED FRAME — only rendered when matched_image is set -->
    {% if matched_image %}
    <section class="frame-card">
      <img src="{{ url_for('result_file', filename=matched_image.filename) }}"
           alt="Matched frame">
    </section>
    {% endif %}

    <!-- FOOTER LINK -->
    <footer class="page-footer">
      <a href="/debug">Developer debug info →</a>
    </footer>
  </div>
  ```

  The form action remains `POST /process`. Keep the same `input[type=file]` and `textarea` as today but remove the help text mentioning JSON intent fields.

  The `answer-card` uses a simple styled box (panel-style background, `--accent` color label). The `frame-card` is a centered image block (max-width: 600px, object-fit: cover). Keep responsive breakpoints.

  The `error` banner is already in the form today — keep it inside the form section.

  Do NOT include: track ID, confidence, bounding boxes, JSON blobs, CSV links, processing stats, runtime/model status, assistant mode string, candidate counts, search scores.

  **Files:** `templates/index.html`

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from app import app; client = app.test_client(); r = client.get('/'); assert r.status_code == 200; assert b'VISTA' in r.data; assert b'track_id' not in r.data; assert b'debug' in r.data.lower(); print('index.html ok')`"

---

- [ ] 7. **Create `templates/debug.html`** — developer page showing all technical detail.

  This page receives two Jinja2 variables: `result` (the full `_last_debug_result` dict or None) and `runtime_status` (dict from `model_status()`).

  Reuse the CSS from the new `index.html` (copy the `<style>` block) and add back the CSS for `.kv`, `.evidence-grid`, `.results-cards`, `.image-meta`, `pre.json`, `.footer-links`, `.status-card` that were removed from `index.html`. It's a standalone template — no inheritance needed.

  ### Sections to render (all inside `{% if result %}` guard):

  1. **Back link** — `← Back to VISTA` linking to `/`
  2. **LLM Runtime status** — render `runtime_status.mode`, `runtime_status.message`, `runtime_status.available`
  3. **Answer** — `result.assistant_message`, `result.assistant_mode`, `result.assistant_error`
  4. **Processing stats** — frames_processed, detections, unique_classes, time_minutes
  5. **Parsed query intent** — `<pre class="json">{{ result.intent | tojson(indent=2) }}</pre>`
  6. **Parser status** — `<pre class="json">{{ result.parse_result | tojson(indent=2) }}</pre>`
  7. **Evidence summary** — all fields from `result.search_result.evidence`: object_class, track_id, first_timestamp, last_timestamp, best_frame, confidence, bbox, related_object, related_track_id, relation, match_score, frames_seen
  8. **Candidate count** — `result.search_result.candidate_count`
  9. **Notes** — `result.search_result.notes` list
  10. **Structured evidence (full JSON)** — `<pre class="json">{{ result.search_result | tojson(indent=2) }}</pre>`
  11. **Visual crops** — all images from `result.images` with filename, class_name, track_id, frame_no, confidence metadata beneath each image
  12. **CSV download links** — `result.detections_csv`, `result.classes_csv` — rendered using `url_for('download_file', filename=...)`
  13. **LLM raw parser output** — `result.parse_result.raw_output` inside a `<pre>` block

  When `result` is None, show a notice: "No results yet — submit a query on the main page first."

  **Files:** `templates/debug.html` (new)

  **Verify:** `cd c:\Users\praan\Desktop\vista_platform && python -c "from app import app; client = app.test_client(); r = client.get('/debug'); assert r.status_code == 200; assert b'debug' in r.data.lower() or b'No results' in r.data; print('debug.html ok')"`

---

- [ ] 8. **Syntax and import verification** — confirm all six Python source files parse cleanly and the Flask app loads.

  Run the following two commands in sequence. Fix any error before completing this step.

  **Command 1 — AST parse all Python files:**
  ```
  cd c:\Users\praan\Desktop\vista_platform && python -c "import ast; [print(f, 'ok') or ast.parse(open(f).read()) for f in ['app.py','llama_service.py','query_parser.py','search_service.py','response_service.py','class_mapper.py']]"
  ```
  Expected: each filename followed by `ok` with no exceptions.

  **Command 2 — Flask app import check:**
  ```
  cd c:\Users\praan\Desktop\vista_platform && python -c "from app import app; print('Flask app imports ok')"
  ```
  Expected: `Flask app imports ok` printed with no `ImportError`, `AttributeError`, or other exceptions.

  If either command fails, read the traceback, identify which file has the error, fix it, and re-run both commands until they pass.

  **Files:** any file flagged by errors above

  **Verify:** both commands above exit without errors.

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Keep `slots=True` off the new dataclasses | `field(default_factory=...)` inside slotted dataclasses requires Python 3.10+; the project doesn't specify a version floor and the YOLO/llama environment may run 3.9. Removing slots is safe — there is no performance-critical path here. |
| `SemanticClassMapper.resolve()` returns only classes in `available_classes` | Prevents the search engine from filtering on a class YOLO never outputs. If `available_classes` is empty (model not loaded yet), the fallback keeps the raw normalized string, so nothing breaks. |
| Fallback when mapper returns empty | If the mapper can't resolve a concept, the search engine falls back to the normalized string directly — identical to the old behaviour — so no regression for known class names like `"person"`. |
| Slim `render_home()` passes only 5 fields | The full `result` dict contained debug data. Passing only `answer_text`, `answer_ok`, `matched_image`, `query`, `error` to the template prevents accidental leakage of technical fields into the main page. |
| `_last_debug_result` module-level variable | Simple, stateless approach sufficient for a single-user demo app. No database or session needed. The variable is populated per `/process` call, so `/debug` always shows the most recent run. |
| `matched_image` is the first image from `result["images"]` | The images list is already sorted by confidence descending in `search_service.finalize()`. The highest-confidence crop is shown on the main page. |
| Debug page is standalone HTML | No Jinja2 template inheritance exists in this project (`index.html` has inline styles). Debug page copies the `<style>` block to stay consistent without adding a base template. |
| Don't change `requirements.txt` | All new code (`class_mapper.py`) uses only the stdlib. No new dependencies. |
