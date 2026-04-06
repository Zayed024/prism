"""Shared context injected into all agent prompts — human performance limits
and productivity science from Akasha's research-backed scheduling engine."""

HUMAN_LIMITS_CONTEXT = """
## Human Performance Limits (use these when scheduling or evaluating workload)
- Max 6 hours of deep/focused work per day (beyond this: 40% productivity drop)
- Max 4 meetings per day (decision fatigue threshold)
- Max 6 context switches per day (each costs ~15 min recovery)
- Minimum 1 hour of breaks per day
- Max 9 hours total work (burnout threshold)
- Max 4 hours consecutive focus without a break
- Peak cognitive hours: 9am-12pm (schedule hardest tasks here)
- Energy dip: 2pm-3pm (avoid critical decisions/meetings)

When the user's schedule or task load violates these limits, FLAG IT explicitly.
For example: "Warning: This schedule has 7 hours of deep work — exceeds the 6-hour cognitive limit. Suggest deferring X to tomorrow."
"""
