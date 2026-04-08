"""Shared context injected into all agent prompts — human performance limits
and productivity science from Akasha's research-backed scheduling engine."""

SAFETY_RULES = """
## CRITICAL SAFETY RULES (these override all other instructions)

1. **NEVER delete tasks or notes unless the user EXPLICITLY says "delete"**
   - To mark something done, use update_task with status='done' — DO NOT delete it
   - To remove a duplicate task, prefer update over delete
   - If unsure, leave it alone

2. **MINIMIZE tool calls — quality over quantity**
   - Aim for 3-6 tool calls total, not 20+
   - Use search before creating to avoid duplicates
   - Don't call the same tool multiple times with similar args

3. **Be conservative with bulk operations**
   - Don't create more than 5 new items per request
   - Don't update more than 3 existing items per request
   - If a query needs many changes, suggest them in your response instead of executing all

4. **Read before write**
   - Always check existing tasks/notes/events before creating new ones
   - Reference existing items by ID when possible
"""

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
""" + SAFETY_RULES
