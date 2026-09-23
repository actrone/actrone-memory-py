"""Claude Agent SDK memory adapter for actrone-memory (Tier 1).

The Claude Agent SDK's ``query(prompt, options=ClaudeAgentOptions(system_prompt=...))`` accepts a
system prompt. This adapter returns the system prompt for a turn, the developer's base system
prompt with governed Actrone memory appended, and persists the completed turn. The SDK has no
formal memory interface, so the system-prompt path is the idiomatic integration.

Framework-free: nothing from ``claude-agent-sdk`` is imported, this runs against a local,
service-free MemoryManager. Mirrors the TypeScript ``claudeAgentMemory`` adapter.
"""

from __future__ import annotations

from actrone_memory.integrations._context import BaseActroneMemory


class ActroneClaudeAgentMemory(BaseActroneMemory):
    """Governed memory for the Claude Agent SDK ``query`` loop.

    Usage::

        from claude_agent_sdk import query, ClaudeAgentOptions
        from actrone_memory.integrations.claude_agent_sdk import ActroneClaudeAgentMemory

        memory = ActroneClaudeAgentMemory(agent_id="support-bot", session_id="s1")
        sys_prompt = await memory.append_to_system_prompt("You are support.", user_input)
        options = ClaudeAgentOptions(system_prompt=sys_prompt)
        async for msg in query(prompt=user_input, options=options):
            ...  # collect the reply into `answer`
        await memory.remember(user_input, answer)
    """

    async def append_to_system_prompt(
        self, base_system_prompt: str, query: str, *, token_budget: int | None = None
    ) -> str:
        """Return the turn's system prompt: governed memory appended to ``base_system_prompt``
        (or just the memory when there is no base system prompt)."""
        context = await self.build_context(query, token_budget=token_budget)
        if not context:
            return base_system_prompt
        return f"{base_system_prompt}\n\n{context}" if base_system_prompt else context
