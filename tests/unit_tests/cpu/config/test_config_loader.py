# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest import mock

import pytest

from torchtitan.config import (
    apply_overrides,
    ConfigLoader,
    Configurable,
    derive,
    override,
    OverrideConfig,
    resolve_root_override,
)
from torchtitan.config.override import _REGISTRY
from torchtitan.config.parallelism import ParallelismConfig


def _noop_post_init() -> None:
    return None


class _Checked(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        bad: bool = False
        value: int = 1

        def __post_init__(self) -> None:
            if self.bad:
                raise ValueError("bad flag")


def recipe_sets_flag_after_init() -> _Checked.Config:
    """Assign after construction, the way seed recipes set their flag."""
    config = _Checked.Config()
    config.bad = True
    return config


def test_loads_exact_recipe_module() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.llama3",
            "--config",
            "llama3_debugmodel",
        ]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"
    assert config.training.steps == 10


def test_loads_recipe_module() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.llama3",
            "--config",
            "llama3_debugmodel",
        ]
    )

    assert type(config.model).__qualname__ == "Llama3Model.Config"


def test_load_uses_current_sys_argv() -> None:
    argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
    with mock.patch.object(sys, "argv", argv):
        with pytest.raises(ImportError, match="Cannot import config module"):
            ConfigLoader().load()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--module", "torchtitan_recipes.tests.models.llama3"],
        ["--config", "llama3_debugmodel"],
    ],
)
def test_module_and_config_are_required(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(args)


def test_unknown_config_lists_available_functions() -> None:
    with pytest.raises(ValueError, match="Available config functions"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
                "--config",
                "not_a_recipe",
            ]
        )


def test_general_config_flags_are_rejected() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
            ]
        )


def test_operational_overrides() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.tests.models.llama3",
            "--config",
            "llama3_debugmodel",
            "--comm-backend",
            "fake",
            "--output-dir",
            "/tmp/torchtitan-test",
        ]
    )

    assert config.comm.backend == "fake"
    assert config.dump_folder == "/tmp/torchtitan-test"


def test_invalid_comm_backend_is_rejected() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--comm-backend",
                "definitely_invalid",
            ]
        )


def test_resume_step_rejects_values_below_latest() -> None:
    with pytest.raises(SystemExit):
        ConfigLoader().load(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--resume-step",
                "-2",
            ]
        )


def test_resume_step_requires_checkpointer() -> None:
    with pytest.raises(ValueError, match="checkpointing"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_recipes.tests.models.llama3",
                "--config",
                "llama3_debugmodel",
                "--resume-step",
                "-1",
            ]
        )


def test_load_revalidates_recipe_assignments() -> None:
    with pytest.raises(ValueError, match="bad flag"):
        ConfigLoader().load(
            [
                "--module",
                __name__,
                "--config",
                "recipe_sets_flag_after_init",
            ]
        )


def test_override_tokens_are_forwarded() -> None:
    config = SimpleNamespace(
        override=OverrideConfig(),
        parallelism=ParallelismConfig(),
        comm=SimpleNamespace(backend="nccl"),
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(
        ConfigLoader, "_trainer_config", return_value=config
    ), mock.patch(
        "torchtitan.config.loader.resolve_root_override", side_effect=lambda cfg: cfg
    ):
        loaded = ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.first",
                "--override",
                'pkg.second={"block_size": 128}',
            ]
        )

    assert loaded is config
    assert config.override.imports == [
        "pkg.first",
        ("pkg.second", {"block_size": 128}),
    ]


def test_recipe_and_cli_overrides_are_applied_together() -> None:
    config = SimpleNamespace(
        override=OverrideConfig(imports=["pkg.recipe"]),
        parallelism=ParallelismConfig(),
        comm=SimpleNamespace(backend="nccl"),
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(
        ConfigLoader, "_trainer_config", return_value=config
    ), mock.patch(
        "torchtitan.config.loader.resolve_root_override", side_effect=lambda cfg: cfg
    ):
        loaded = ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.cli",
            ]
        )

    assert loaded is config
    assert config.override.imports == ["pkg.recipe", "pkg.cli"]


def test_cli_overrides_extend_the_top_level_rl_config() -> None:
    trainer = SimpleNamespace(comm=SimpleNamespace(backend="nccl"))
    config = SimpleNamespace(
        override=OverrideConfig(imports=["pkg.recipe"]),
        trainer=trainer,
        dump_folder="outputs",
        hf_assets_path=None,
        __post_init__=_noop_post_init,
    )
    with mock.patch.object(
        ConfigLoader, "_load_config", return_value=config
    ), mock.patch.object(
        ConfigLoader, "_trainer_config", return_value=trainer
    ), mock.patch(
        "torchtitan.config.loader.resolve_root_override", side_effect=lambda cfg: cfg
    ):
        ConfigLoader().load(
            [
                "--module",
                "custom",
                "--config",
                "recipe",
                "--override",
                "pkg.cli",
            ]
        )

    assert config.override.imports == ["pkg.recipe", "pkg.cli"]


class _Launch(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        child: _Checked.Config = field(default_factory=_Checked.Config)
        override: OverrideConfig = field(default_factory=OverrideConfig)
        dump_folder: str = "outputs"

    def __init__(self, config):
        apply_overrides(config.override, config)
        self.config = config


class _AlternateLaunch(_Launch):
    @dataclass(kw_only=True, slots=True)
    class Config(_Launch.Config):
        multiplier: int = 2


@pytest.fixture
def root_overrides():
    saved = dict(_REGISTRY)
    calls = []

    @override(target=_Launch.Config, fqns=[""])
    def select_launch(config, *, multiplier=2):
        calls.append(config.dump_folder)
        config.child.value += 1
        return derive(config, _AlternateLaunch.Config, multiplier=multiplier)

    @override(target=_Checked.Config)
    def increment_child(config):
        return derive(config, _Checked.Config, value=config.value + 10)

    try:
        yield f"{__name__}.select_launch", f"{__name__}.increment_child", calls
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved)


@pytest.mark.parametrize("root_first", [True, False])
def test_root_override_selects_owner_before_build(root_overrides, root_first):
    root, child, calls = root_overrides
    source = _Launch.Config()
    entries = [f'{root}={{"multiplier": 3}}', child]
    if not root_first:
        entries.reverse()
    args = ["--module", "unused", "--config", "unused", "--output-dir", "selected"]
    for entry in entries:
        args.extend(["--override", entry])
    with mock.patch.object(ConfigLoader, "_load_config", return_value=source):
        config = ConfigLoader().load(args)
    assert calls == ["selected"]
    assert config.child.value == 2  # Component replacement still waits for the owner.
    assert config.override.imports == [child]
    assert source.child.value == 1  # A root factory cannot mutate the recipe's tree.
    assert len(source.override.imports) == 2
    assert resolve_root_override(config) is config
    instance = config.build()
    assert isinstance(instance, _AlternateLaunch)
    assert instance.config.multiplier == 3
    assert instance.config.child.value == 12  # Applied exactly once, at construction.
    assert calls == ["selected"]


def test_component_only_loading_preserves_owner_and_timing(root_overrides):
    _, child, calls = root_overrides
    source = _Launch.Config(override=OverrideConfig(imports=[child]))
    with mock.patch.object(ConfigLoader, "_load_config", return_value=source):
        config = ConfigLoader().load(["--module", "unused", "--config", "unused"])
    assert config is source
    assert config.child.value == 1
    assert calls == []
    instance = config.build()
    assert type(instance) is _Launch
    assert instance.config.child.value == 11


def test_root_conflicts_fail_before_either_factory_runs(root_overrides):
    root, _, calls = root_overrides

    @override(target=_Launch.Config, fqns=[""])
    def another_launch(config):
        raise AssertionError("Conflicting factories must not run")

    config = _Launch.Config(
        override=OverrideConfig(imports=[root, f"{__name__}.another_launch"])
    )
    with pytest.raises(ValueError, match="Multiple overrides claim the root"):
        resolve_root_override(config)
    assert calls == []


@pytest.mark.parametrize("exact,fqns", [(True, [""]), (False, ["child"])])
def test_root_selection_respects_exact_type_and_fqns(root_overrides, exact, fqns):
    @override(target=_Launch.Config, exact=exact, fqns=fqns)
    def unselected(config):
        raise AssertionError("This override must not match the root")

    config = _AlternateLaunch.Config(
        override=OverrideConfig(imports=[f"{__name__}.unselected"])
    )
    assert resolve_root_override(config) is config


def test_python_root_selection_revalidates_replacement(root_overrides):
    @override(target=_Launch.Config, fqns=[""])
    def invalid_root(config):
        replacement = _Checked.Config()
        replacement.bad = True
        return replacement

    config = _Launch.Config(
        override=OverrideConfig(imports=[f"{__name__}.invalid_root"])
    )
    with pytest.raises(ValueError, match="bad flag"):
        resolve_root_override(config)
