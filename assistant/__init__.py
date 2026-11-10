"""`assistant` is a chat that plans an experiment and fills in its setup.

Tools (assistant/tools/) act as the person through `assistant.gateway`, which
denies them the `access.policy.HUMAN_ONLY` actions. The model is any
OpenAI-compatible chat server. Marked files adapt code from A3S
(https://github.com/LUH-AI/A3S), BSD-3-Clause, Copyright (c) Marcel Wever.
"""
