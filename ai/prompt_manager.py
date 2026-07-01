"""Prompt template management and message assembly.

Uses ``string.Template`` — no external dependency.
"""

from __future__ import annotations

from string import Template
from typing import Any

from ai.models import Message, Role


class PromptManager:
    """Manage prompt templates and build message lists.

    Usage::

        pm = PromptManager()
        pm.add_template("summarise", "Summarise: $text")
        prompt = pm.render("summarise", text="Hello world")
        messages = pm.build_messages(
            system_prompt="You are a helpful assistant.",
            user_prompt=prompt,
        )
    """

    def __init__(self) -> None:
        self._templates: dict[str, Template] = {}

    def add_template(self, name: str, template_str: str) -> None:
        """Register a named prompt template.

        Args:
            name: Unique template identifier.
            template_str: Template string using ``$var`` placeholders.
        """
        self._templates[name] = Template(template_str)

    def remove_template(self, name: str) -> None:
        """Remove a previously registered template."""
        self._templates.pop(name, None)

    def render(self, template_name: str, **variables: Any) -> str:
        """Render a named template with the given variables.

        Args:
            template_name: Template identifier registered via ``add_template``.
            **variables: Values for template placeholders.

        Returns:
            The rendered string.

        Raises:
            KeyError: If the template name does not exist.
        """
        template = self._templates.get(template_name)
        if template is None:
            msg = f"Unknown template: '{template_name}'"
            raise KeyError(msg)
        return template.safe_substitute(**variables)

    def build_messages(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        history: list[Message] | None = None,
    ) -> list[dict[str, str]]:
        """Build an OpenAI-style message list.

        Args:
            system_prompt: Optional system-level instruction.
            user_prompt: The current user message.
            history: Prior conversation messages (alternating user/assistant).

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": Role.SYSTEM.value, "content": system_prompt})

        if history:
            for msg in history:
                messages.append({"role": msg.role.value, "content": msg.content})

        if user_prompt:
            messages.append({"role": Role.USER.value, "content": user_prompt})

        return messages

    def list_templates(self) -> list[str]:
        """Return names of all registered templates."""
        return list(self._templates.keys())
