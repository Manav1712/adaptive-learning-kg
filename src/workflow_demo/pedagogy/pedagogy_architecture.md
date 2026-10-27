# Pedagogy runtime boundary

The canonical design write-up is [`docs/ADAPTIVE_TUTORING_ARCHITECTURE_SPEC.md`](../../../docs/ADAPTIVE_TUTORING_ARCHITECTURE_SPEC.md). This note only says what runs by default.

## Default path

On each tutor turn with student text, `BotSessionManager` runs:

- session and turn progression
- `MisconceptionDiagnoser`
- `TeachingMoveGenerator`
- `PolicyScorer`
- `derive_instruction_lo`
- `PedagogicalRetrievalPolicy`
- then `tutor_bot`

`TeachingMoveType.EXPLAIN_CONCEPT` stays in the enum. Progression and the tutor prompt know the name. The generator does not emit it.

## Off unless a flag is set

Flags live in `src/workflow_demo/practice/feature_flags.py`. All default off.

- `WORKFLOW_DEMO_ENABLE_PRACTICE_LOOP=1` — practice loop
- that flag, plus `WORKFLOW_DEMO_ENABLE_ADAPTIVE_SEQUENCING=1` and `WORKFLOW_DEMO_SEQUENCER_MODE=heuristic` — heuristic sequencer
- the same, with `WORKFLOW_DEMO_SEQUENCER_MODE=pomdp` — POMDP sequencer, MPC, and the particle filter

`llm_attempt_observer_enabled` (`WORKFLOW_DEMO_ENABLE_LLM_ATTEMPT_OBSERVER`) is parsed into `PracticeFeatureFlags` and is not read by the tutor.
