"""Utilities for working with the persona profile data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(slots=True)
class PersonaProfile:
    """Structured representation of the persona information."""

    name: str
    tagline: str
    bio: str
    highlights: List[str]
    expertise: Dict[str, List[str]]
    projects: List[Dict[str, Any]]
    fun_facts: List[str]

    @property
    def summary(self) -> str:
        highlight_lines = "\n".join(f"- {item}" for item in self.highlights)
        expertise_lines = "\n".join(
            f"- {domain}: {', '.join(items)}" for domain, items in self.expertise.items()
        )
        project_lines = "\n".join(
            f"- {p['name']} ({p['role']}): {p['impact']}" for p in self.projects
        )
        fun_lines = "\n".join(f"- {fact}" for fact in self.fun_facts)
        return (
            f"Name: {self.name}\n"
            f"Tagline: {self.tagline}\n"
            f"Bio: {self.bio}\n\n"
            f"Highlights:\n{highlight_lines}\n\n"
            f"Expertise:\n{expertise_lines}\n\n"
            f"Projects:\n{project_lines}\n\n"
            f"Fun facts:\n{fun_lines}"
        )


def load_profile(path: Path) -> PersonaProfile:
    """Load persona information from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return PersonaProfile(
        name=data["name"],
        tagline=data["tagline"],
        bio=data["bio"],
        highlights=data.get("highlights", []),
        expertise=data.get("expertise", {}),
        projects=data.get("projects", []),
        fun_facts=data.get("fun_facts", []),
    )


def search_profile(profile: PersonaProfile, query: str) -> str:
    """Return a focused snippet of the persona based on a simple keyword search."""

    query_lower = query.lower()
    sections: list[str] = []

    if any(keyword in query_lower for keyword in ("who", "bio", "background")):
        sections.append(profile.bio)

    if any(keyword in query_lower for keyword in ("skill", "expert", "tech")):
        expertise_lines = "\n".join(
            f"- {domain}: {', '.join(items)}" for domain, items in profile.expertise.items()
        )
        sections.append(f"Key strengths:\n{expertise_lines}")

    if any(keyword in query_lower for keyword in ("project", "work", "built")):
        project_lines = "\n".join(
            f"- {p['name']} ({p['role']}): {p['impact']}" for p in profile.projects
        )
        sections.append(f"Notable projects:\n{project_lines}")

    if any(keyword in query_lower for keyword in ("fun", "hobby", "interest")):
        fun_lines = "\n".join(f"- {fact}" for fact in profile.fun_facts)
        sections.append(f"Fun facts:\n{fun_lines}")

    if not sections:
        sections.append(profile.summary)

    return "\n\n".join(sections)

