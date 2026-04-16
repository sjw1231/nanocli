"""Unit tests for HW3 skill discovery and implicit/explicit skill context."""

from pathlib import Path

from nanocli.skill import (
    Skill,
    discover_skills,
    load_skills,
    resolve_explicit_skill_injections,
)


def _write_skill_file(
    path: Path,
    *,
    name: str,
    description: str,
    body: str = "# Skill body",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ("---\n" f"name: {name}\n" f"description: {description}\n" "---\n\n" f"{body}\n"),
        encoding="utf-8",
    )
    return path.resolve()


def test_discover_skills_parses_valid_frontmatter(tmp_path: Path) -> None:
    path = _write_skill_file(
        tmp_path / "skills" / "alpha-skill" / "SKILL.md",
        name="alpha-skill",
        description="Alpha description",
    )

    discovered = discover_skills([tmp_path / "skills"])

    assert len(discovered) == 1
    assert discovered[0].name == "alpha-skill"
    assert discovered[0].description == "Alpha description"
    assert discovered[0].path == path
    assert discovered[0].contents.startswith("---\n")


def test_discover_skills_skips_missing_frontmatter(tmp_path: Path) -> None:
    invalid = tmp_path / "skills" / "broken" / "SKILL.md"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("name: broken\ndescription: no delimiters\n", encoding="utf-8")

    assert discover_skills([tmp_path / "skills"]) == []


def test_discover_skills_skips_missing_required_frontmatter_fields(tmp_path: Path) -> None:
    invalid = tmp_path / "skills" / "broken" / "SKILL.md"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("---\nname: broken\n---\n", encoding="utf-8")

    assert discover_skills([tmp_path / "skills"]) == []


def test_discover_skills_skips_invalid_frontmatter_line(tmp_path: Path) -> None:
    invalid = tmp_path / "skills" / "broken" / "SKILL.md"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text("---\nname broken\ndescription: value\n---\n", encoding="utf-8")

    assert discover_skills([tmp_path / "skills"]) == []


def test_discover_skills_across_roots_dedupes_and_parent_loader_sorts(tmp_path: Path) -> None:
    root_a = tmp_path / "skills_a"
    root_b = tmp_path / "skills_b"
    beta = _write_skill_file(
        root_a / "beta-skill" / "SKILL.md",
        name="beta-skill",
        description="Beta",
    )
    alpha = _write_skill_file(
        root_b / "alpha-skill" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
    )

    discovered = discover_skills([root_a, root_b, root_a])
    assert {(skill.name, skill.path) for skill in discovered} == {
        ("alpha-skill", alpha),
        ("beta-skill", beta),
    }

    ordered = load_skills([root_a, root_b, root_a])
    assert [skill.name for skill in ordered] == ["alpha-skill", "beta-skill"]
    assert [skill.path for skill in ordered] == [alpha, beta]


def test_skill_render_explicit_exact_envelope_format(tmp_path: Path) -> None:
    path = _write_skill_file(
        tmp_path / "skills" / "alpha" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
        body="# Body",
    )
    skill = Skill(
        name="alpha-skill",
        description="Alpha",
        path=path,
        contents="---\nname: alpha-skill\ndescription: Alpha\n---\n\n# Body\n",
    )

    assert skill.render_explicit() == (
        "<skill>\n"
        "<name>alpha-skill</name>\n"
        f"<path>{path}</path>\n"
        "---\nname: alpha-skill\ndescription: Alpha\n---\n\n# Body\n\n"
        "</skill>"
    )


def test_resolve_explicit_skill_injections_returns_selected_skill(tmp_path: Path) -> None:
    path = _write_skill_file(
        tmp_path / "skills" / "alpha" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
        body="# Body",
    )
    selected = resolve_explicit_skill_injections(
        "$alpha-skill please help",
        load_skills([tmp_path / "skills"]),
    )

    assert selected is not None
    assert selected == Skill(
        name="alpha-skill",
        description="Alpha",
        path=path,
        contents="---\nname: alpha-skill\ndescription: Alpha\n---\n\n# Body\n",
    )

    assert "<name>alpha-skill</name>" in selected.render_explicit()


def test_resolve_explicit_skill_injections_rejects_invalid_name_token(tmp_path: Path) -> None:
    _write_skill_file(
        tmp_path / "skills" / "alpha" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
    )
    loaded = load_skills([tmp_path / "skills"])
    assert resolve_explicit_skill_injections("$alpha! run", loaded) is None


def test_resolve_explicit_skill_injections_accepts_bare_prefix_token(tmp_path: Path) -> None:
    _write_skill_file(
        tmp_path / "skills" / "alpha" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
    )
    loaded = load_skills([tmp_path / "skills"])
    selected = resolve_explicit_skill_injections("$alpha-skill", loaded)
    assert selected is not None
    assert selected.name == "alpha-skill"
    assert resolve_explicit_skill_injections("please use $alpha-skill now", loaded) is None


def test_resolve_explicit_skill_injections_rejects_ambiguous_plain_name(tmp_path: Path) -> None:
    path_a = _write_skill_file(
        tmp_path / "skills" / "demo-a" / "SKILL.md",
        name="demo-skill",
        description="A",
    )
    path_b = _write_skill_file(
        tmp_path / "skills" / "demo-b" / "SKILL.md",
        name="demo-skill",
        description="B",
    )
    assert path_a != path_b
    assert resolve_explicit_skill_injections("$demo-skill do work", load_skills([tmp_path / "skills"])) is None


def test_resolve_explicit_skill_injections_returns_none_for_unknown_name(tmp_path: Path) -> None:
    _write_skill_file(
        tmp_path / "skills" / "alpha" / "SKILL.md",
        name="alpha-skill",
        description="Alpha",
    )
    assert resolve_explicit_skill_injections("$beta-skill do work", load_skills([tmp_path / "skills"])) is None
