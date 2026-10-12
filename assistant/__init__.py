"""The assistant: a chat that plans an experiment with the person, in plain
language, and fills in its setup for them.

It acts only through tools (assistant/tools/), every one of which reaches the
application through the gateway (assistant/gateway.py), as the person and never
as more than them: the policy refuses an assistant what the person must do
themselves (`ui.permissions.HUMAN_ONLY`).

The model behind it is any server that speaks the OpenAI chat API
(assistant/llm.py). Parts of this app are ported from A3S
(https://github.com/LUH-AI/A3S), BSD-3-Clause, Copyright (c) Marcel Wever; the
files that carry its code say so.
"""
