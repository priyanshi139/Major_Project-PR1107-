# HealthTwin vs the DiabeTwin reference project

DiabeTwin (https://github.com/kalashjain9/diabetwin) was used only as a reference for interface style and Digital Twin interaction ideas. No DiabeTwin code was copied; the HealthTwin interface, CSS and modules were written independently.

## What was actually inspected (and what was not)
Inspected: the repository README (concept, architecture diagram, folder layout, claims) and the first ~700 lines of `app.py` (page CSS, session state, profile-creation form, sample profiles, metric cards, profile card, start of the "current status" tab, and the imports/usage of its `core`, `genai` and `ui` packages).
NOT inspected: the source files inside `core/`, `genai/` and `ui/` could not be opened with the available tools, and the remainder of `app.py` (about 1,000 further lines) was not read. Statements about those internals below are inferred from README text and from how `app.py` calls them, and are marked as such.

## What DiabeTwin appears to implement
| Idea | Evidence | Status |
|---|---|---|
| Streamlit app with custom gradient CSS, metric cards, profile card, tabs | `app.py` code read directly | Implemented |
| Virtual patient from a form or sample profiles (4 synthetic profiles) | `app.py` code read directly | Implemented, synthetic |
| Health memory (snapshots of past states) seeded by simulating 6 months | `initialize_health_memory()` read directly | Implemented, simulated |
| Temporal evolution engine with scenarios (`simulate_trajectory(patient, months, scenario_type)`) | usage seen in `app.py`; engine source not read | Inferred to exist |
| Narrative engine with `fallback_mode=True`, counterfactual story generator | usage seen in `app.py`; source not read | Inferred; README says it works without an API key using templates |
| Lifestyle sliders, trajectory charts, chat interface | README + imports | README-described; implementation not read |
| Data type | README states synthetic data and simplified models | Not clinical |

Its health evolution is a simulation over synthetic patients (diabetes-specific), not a model trained on real patient data.

## What HealthTwin does differently
| Aspect | DiabeTwin (as inspected) | HealthTwin |
|---|---|---|
| Data | Synthetic profiles | Real historical public data (PhysioNet 2019 sepsis; MIT-BIH ECG) + clearly labelled synthetic + user uploads |
| Models | Simplified simulation | Trained, evaluated ML models with leakage-safe splits and honest metrics |
| Twin state | Simulated trajectory in months | Hourly twin updated by replayed observations; no future data used |
| What-if | Lifestyle simulation | Model sensitivity analysis on observed history (explicitly not a treatment simulation) |
| Alerts | n/a in inspected code | Severity, cooldown, escalation workflow with human oversight and audit trail |
| Explanations | Template/LLM narrative | Rule-based text restating computed numbers; linear-model factors only |

## Adopted from the reference idea (re-implemented)
Sidebar navigation + card-based dashboard, patient profile card, health memory (here: `state_at` slider replaying earlier twin states), what-if scenarios, template-based insights that work without any API key.

## Deliberately not adopted
Lifestyle-driven disease progression simulation for diabetes, hypertension or cardiovascular disease: no verified dataset or validated model exists in this project, so those conditions are shown as "not implemented" instead of being simulated.
