"""Removed no-op config keys: accepted, dropped, and reported with one DeprecationWarning.

RHEN 1.0.0 deleted 21 config knobs that no code read. Configs, presets and saved
checkpoints written before that still carry them, so every constructor must keep
accepting them. Each call that receives removed keys drops them, emits exactly one
DeprecationWarning naming them, and leaves no attribute behind.
"""

import glob
import json
import os
import tempfile
import unittest
import warnings

from config import (
    ActBudgetConfig,
    AttentionMLAConfig,
    EOSConfig,
    LoopControlConfig,
    MoEConfig,
    OrchestratorConfig,
    RecurrentLoopConfig,
    RhenEngineConfig,
    YaRNConfig,
)
from eos_research.presets import preset_config
from eos_research.train import TrainConfig

ROOT = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == "tests"
    else os.path.dirname(os.path.abspath(__file__))
)

#: (group class that owned the key, key, value an old config would carry).
LEGACY = [
    (RecurrentLoopConfig, "enable_symplectic_loop", True),
    (AttentionMLAConfig, "use_flash_attn", True),
    (AttentionMLAConfig, "diff_attn_lambda_init", 0.8),
    (AttentionMLAConfig, "enable_prefix_lm_mask", True),
    (YaRNConfig, "yarn_scale_factor", 1.0),
    (RhenEngineConfig, "rhen_calib_loss_coef", 0.05),
    (RhenEngineConfig, "rhen_enable_deep_reasoning", True),
    (RhenEngineConfig, "rhen_confidence_threshold", 0.75),
    (RhenEngineConfig, "rhen_enable_qwen_reasoning", True),
    (RhenEngineConfig, "rhen_enable_deepseek_reasoning", True),
    (RhenEngineConfig, "rhen_meta_reasoning_cache_size", 1000),
    (RhenEngineConfig, "rhen_enable_three_layer_pipeline", True),
    (RhenEngineConfig, "rhen_semantic_layer_weight", 0.4),
    (RhenEngineConfig, "rhen_reasoning_layer_weight", 0.35),
    (RhenEngineConfig, "rhen_generation_layer_weight", 0.25),
    (RhenEngineConfig, "rhen_enable_confidence_routing", True),
    (RhenEngineConfig, "rhen_min_semantic_confidence", 0.85),
    (RhenEngineConfig, "rhen_max_reasoning_depth", 3),
    (OrchestratorConfig, "use_fused_recurrent_kernels", False),
    (OrchestratorConfig, "checkpoint_policy", "mla,moera"),
    (OrchestratorConfig, "use_triton", True),
]
LEGACY_VALUES = {key: value for _cls, key, value in LEGACY}


def _capture(fn):
    """Run ``fn`` and return (its result, the removed-key DeprecationWarnings it emitted)."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = fn()
    removed = [
        w for w in record
        if issubclass(w.category, DeprecationWarning) and "removed no-op config key" in str(w.message)
    ]
    return result, removed


class TestLegacyKeysOnFlatConstructor(unittest.TestCase):
    def test_each_removed_key_is_accepted_warned_and_dropped(self):
        for _cls, key, value in LEGACY:
            with self.subTest(key=key):
                cfg, removed = _capture(lambda: EOSConfig(**{key: value}))
                self.assertEqual(len(removed), 1, [str(w.message) for w in removed])
                self.assertIn(key, str(removed[0].message))
                self.assertFalse(hasattr(cfg, key))
                self.assertNotIn(key, cfg.to_dict())

    def test_all_removed_keys_in_one_call_give_one_warning_naming_all(self):
        cfg, removed = _capture(lambda: EOSConfig(hidden_size=32, **LEGACY_VALUES))
        self.assertEqual(len(removed), 1)
        message = str(removed[0].message)
        for key in LEGACY_VALUES:
            self.assertIn(key, message)
        self.assertEqual(cfg.hidden_size, 32)
        self.assertFalse(any(hasattr(cfg, key) for key in LEGACY_VALUES))

    def test_warning_points_at_the_caller(self):
        _cfg, removed = _capture(lambda: EOSConfig(use_triton=True))
        self.assertEqual(os.path.abspath(removed[0].filename), os.path.abspath(__file__))

    def test_no_warning_without_removed_keys(self):
        cfg, removed = _capture(lambda: EOSConfig(hidden_size=32))
        self.assertEqual(removed, [])
        self.assertEqual(cfg.hidden_size, 32)

    def test_remaining_flat_kwargs_still_apply(self):
        cfg, _removed = _capture(lambda: EOSConfig(use_triton=False, rhen_num_slots=4, use_moe=False))
        self.assertEqual(cfg.rhen_num_slots, 4)
        self.assertFalse(cfg.use_moe)


class TestLegacyKeysOnFromDict(unittest.TestCase):
    def test_dict_with_every_removed_key_loads_with_one_warning(self):
        payload = dict(LEGACY_VALUES, hidden_size=32, intermediate_size=64)
        original = dict(payload)
        cfg, removed = _capture(lambda: EOSConfig.from_dict(payload))
        self.assertEqual(len(removed), 1)
        for key in LEGACY_VALUES:
            self.assertIn(key, str(removed[0].message))
            self.assertFalse(hasattr(cfg, key))
            self.assertNotIn(key, cfg.to_dict())
        self.assertEqual(cfg.hidden_size, 32)
        self.assertEqual(payload, original, "the caller's dict must not be mutated")

    def test_each_removed_key_alone_from_dict(self):
        for _cls, key, value in LEGACY:
            with self.subTest(key=key):
                cfg, removed = _capture(lambda: EOSConfig.from_dict({key: value}))
                self.assertEqual(len(removed), 1)
                self.assertIn(key, str(removed[0].message))
                self.assertFalse(hasattr(cfg, key))

    def test_removed_key_passed_as_from_dict_kwarg_is_also_warned(self):
        cfg, removed = _capture(lambda: EOSConfig.from_dict({"hidden_size": 32}, use_triton=True))
        self.assertEqual(len(removed), 1)
        self.assertIn("use_triton", str(removed[0].message))
        self.assertFalse(hasattr(cfg, "use_triton"))

    def test_clean_dict_emits_no_removed_key_warning(self):
        cfg, removed = _capture(lambda: EOSConfig.from_dict({"hidden_size": 32}))
        self.assertEqual(removed, [])
        self.assertEqual(cfg.hidden_size, 32)

    def test_legacy_checkpoint_config_json_loads(self):
        """A config.json saved before the removal, with the old keys added back, still loads."""
        with tempfile.TemporaryDirectory() as tmp:
            current = EOSConfig(hidden_size=32, intermediate_size=64)
            current.save_pretrained(tmp)
            path = os.path.join(tmp, "config.json")
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            saved.update(LEGACY_VALUES)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(saved, fh)

            loaded, removed = _capture(lambda: EOSConfig.from_pretrained(tmp))
        self.assertEqual(len(removed), 1)
        self.assertEqual(loaded.hidden_size, 32)
        self.assertEqual(loaded.to_dict().keys() & LEGACY_VALUES.keys(), set())


class TestLegacyKeysOnGroupObjects(unittest.TestCase):
    def test_group_constructor_accepts_removed_key_with_warning(self):
        for group_cls, key, value in LEGACY:
            with self.subTest(key=key):
                group, removed = _capture(lambda: group_cls(**{key: value}))
                self.assertEqual(len(removed), 1)
                self.assertIn(key, str(removed[0].message))
                self.assertFalse(hasattr(group, key))
                # The removed key does not disturb any real field.
                self.assertEqual(group, group_cls())
                self.assertEqual(os.path.abspath(removed[0].filename), os.path.abspath(__file__))

    def test_nested_group_with_removed_key_flows_into_eosconfig(self):
        group, removed = _capture(lambda: RhenEngineConfig(rhen_confidence_threshold=0.1, rhen_num_slots=2))
        self.assertEqual(len(removed), 1)
        self.assertEqual(group.rhen_num_slots, 2)

        cfg, removed = _capture(lambda: EOSConfig(rhen_engine=group))
        self.assertEqual(removed, [])
        self.assertEqual(cfg.rhen_num_slots, 2)
        self.assertFalse(hasattr(cfg, "rhen_confidence_threshold"))

    def test_untouched_groups_keep_their_defaults(self):
        _cfg, removed = _capture(lambda: EOSConfig(yarn=YaRNConfig(yarn_scale_factor=2.0), hidden_size=32))
        self.assertEqual(len(removed), 1)
        self.assertEqual(EOSConfig(hidden_size=32).yarn, YaRNConfig())

    def test_groups_without_removed_keys_construct_silently(self):
        for group_cls in (
            RecurrentLoopConfig, MoEConfig, AttentionMLAConfig, YaRNConfig,
            ActBudgetConfig, RhenEngineConfig, LoopControlConfig, OrchestratorConfig,
        ):
            with self.subTest(group=group_cls.__name__):
                _group, removed = _capture(group_cls)
                self.assertEqual(removed, [])


class TestShippedConfigsStillLoad(unittest.TestCase):
    def test_every_config_json_loads(self):
        paths = sorted(glob.glob(os.path.join(ROOT, "configs", "*.json")))
        self.assertTrue(paths, "no configs/*.json found")
        for path in paths:
            with self.subTest(config=os.path.basename(path)):
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                cfg, removed = _capture(lambda: TrainConfig.from_dict(data))
                cfg.validate()
                self.assertEqual(removed, [])
                # The run config's preset must still build a model config without removed keys.
                model_kwargs = preset_config(cfg.preset, **cfg.model_overrides)
                model_cfg, removed = _capture(lambda: EOSConfig(**model_kwargs))
                self.assertEqual(removed, [])
                self.assertIsInstance(model_cfg, EOSConfig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
