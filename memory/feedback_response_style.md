---
name: Response style preferences
description: How the user wants recommendations and explanations delivered
type: feedback
---

When the user asks "which one should I use" or compares approaches, give ONE direct recommendation with reasoning, not a balanced menu.

**Why**: User asked "mediapipe or detectron2 which one" after already getting a comparison — repeating tradeoffs frustrates them. They want the call made, then they decide.

**How to apply**:
- Lead with the recommendation in bold
- Follow with 3-5 numbered reasons tied to *their* specific situation (their hardware, their data, their prior decisions)
- End with one concrete next-step question ("Want me to wire it up?")
- Avoid hedging language like "it depends" or "both are valid"
- If the answer truly depends on a missing fact, ask for that fact instead of listing both options
