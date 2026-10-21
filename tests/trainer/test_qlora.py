import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestQloraExports(unittest.TestCase):
    def test_resolve_training_export_format_auto_for_vllm(self) -> None:
        from afterlife.trainer.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format("vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
            "lora",
        )

    def test_resolve_training_export_format_auto_for_llama(self) -> None:
        from afterlife.trainer.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
            ),
            "gguf",
        )

    def test_resolve_training_export_format_respects_llama_gguf_override(self) -> None:
        from afterlife.trainer.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "llama:unsloth/Qwen2.5-Coder-7B-Instruct-GGUF",
                configured_format="gguf",
            ),
            "gguf",
        )

    @patch("afterlife.trainer.qlora.logger.warning")
    def test_resolve_training_export_format_warns_for_vllm_merged_override(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from afterlife.trainer.qlora import resolve_training_export_format

        self.assertEqual(
            resolve_training_export_format(
                "vllm:Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
                configured_format="merged_16bit",
            ),
            "merged_16bit",
        )
        mock_warning.assert_called_once()


class TestLoadSftDataset(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_tokenizer(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                user_content = messages[0]["content"]
                if len(messages) == 1:
                    return f"<user>{user_content}</user><asst>"
                asst_content = messages[1]["content"]
                return f"<user>{user_content}</user><asst>{asst_content}</asst>"

            def __call__(self, text=None, **_kwargs):
                assert isinstance(text, str)
                return {
                    "input_ids": [ord(char) for char in text],
                    "attention_mask": [1] * len(text),
                }

        return FakeTokenizer()

    def _make_row(self, *, success: bool = True, **kwargs) -> dict:
        base = {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": success,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_score": 100.0,
        }
        base.update(kwargs)
        return base

    def test_load_sft_dataset_from_jsonl(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], path)
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertGreater(len(dataset), 0)
        self.assertIn("input_ids", dataset.column_names)
        self.assertIn("completion_mask", dataset.column_names)
        self.assertNotIn("text", dataset.column_names)
        self.assertNotIn("prompt", dataset.column_names)
        self.assertNotIn("completion", dataset.column_names)

    def test_load_sft_dataset_templates_and_masks_response(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(prompt="Question", raw_response="Answer")], path
            )
            dataset = _load_sft_dataset([path], tokenizer)

        full_text = "<user>Question</user><asst>Answer</asst>"
        prompt_text = "<user>Question</user><asst>"
        self.assertEqual(dataset[0]["input_ids"], [ord(char) for char in full_text])
        self.assertEqual(
            dataset[0]["completion_mask"][: len(prompt_text)],
            [0] * len(prompt_text),
        )
        self.assertEqual(
            dataset[0]["completion_mask"][len(prompt_text) :],
            [1] * (len(full_text) - len(prompt_text)),
        )

    def test_load_sft_dataset_drops_rows_with_fully_truncated_response(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            # FakeTokenizer: input_ids = [ord(c) for c in text].
            # short prompt -> prompt prefix "<user>hi</user><asst>" = 21 tokens (< 40),
            #   response tokens survive the max_seq_length=40 window -> KEPT.
            short = self._make_row(prompt="hi", raw_response="ok")
            # long prompt -> prompt prefix "<user>" + 100 + "</user><asst>" = 119 tokens
            #   (>= 40), so completion_mask[:40] is all zeros -> response fully
            #   truncated -> all-(-100) labels -> nan-loss row -> DROPPED.
            long_prompt = self._make_row(prompt="x" * 100, raw_response="answer")
            self._make_jsonl([short, long_prompt], path)
            dataset = _load_sft_dataset([path], tokenizer, max_seq_length=40)

        self.assertEqual(len(dataset), 1)
        kept = "".join(chr(token_id) for token_id in dataset[0]["input_ids"])
        self.assertIn("hi", kept)
        self.assertIn(1, dataset[0]["completion_mask"][:40])

    def test_completion_survives_truncation_helper(self):
        from afterlife.trainer.qlora import _completion_survives_truncation

        self.assertTrue(
            _completion_survives_truncation(
                {"completion_mask": [0, 1]},
                max_seq_length=2,
            )
        )
        self.assertFalse(
            _completion_survives_truncation(
                {"completion_mask": [0, 0, 1]},
                max_seq_length=2,
            )
        )
        self.assertTrue(
            _completion_survives_truncation(
                {"completion_mask": [0, 0, 1]},
                max_seq_length=None,
            )
        )

    def test_load_sft_dataset_drops_rehearsal_rows_with_fully_truncated_response(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(prompt="hi", raw_response="ok")], path)

            with patch(
                "afterlife.trainer.qlora.load_dataset",
                return_value=[{"text": "x" * 120}],
            ):
                dataset = _load_sft_dataset(
                    [path],
                    tokenizer,
                    max_seq_length=40,
                    rehearsal_dataset="demo/rehearsal",
                    rehearsal_rows_per_epoch=1,
                    rehearsal_prompt_chars=100,
                    rehearsal_max_chars=120,
                )

        self.assertEqual(len(dataset), 1)
        kept = "".join(chr(token_id) for token_id in dataset[0]["input_ids"])
        self.assertIn("hi", kept)
        self.assertNotIn("Continue the following", kept)

    def test_load_sft_dataset_keeps_all_rows_when_max_seq_length_unset(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [
                    self._make_row(prompt="hi", raw_response="ok"),
                    self._make_row(prompt="x" * 100, raw_response="answer"),
                ],
                path,
            )
            # No max_seq_length -> no truncation-based dropping (default behaviour).
            dataset = _load_sft_dataset([path], tokenizer)

        self.assertEqual(len(dataset), 2)

    def test_load_sft_dataset_adds_length_column_for_grouping(self):
        # group_by_length needs a per-row length; expose it as a "length" column
        # (TrainingArguments.length_column_name default) so the sampler groups
        # similar-length rows without re-tokenizing.
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(prompt="Question", raw_response="Answer")], path
            )
            dataset = _load_sft_dataset([path], tokenizer)

        self.assertIn("length", dataset.column_names)
        self.assertEqual(dataset[0]["length"], len(dataset[0]["input_ids"]))

    def test_build_query_response_sft_row_rejects_non_prefix_template(self):
        # Fallback path (response text not found in the rendering): the NSL2
        # boundary check still rejects templates whose prompt is not a prefix.
        from afterlife.trainer.qlora import _build_query_response_sft_row

        tokenizer = self._make_tokenizer()
        original = tokenizer.apply_chat_template

        def bad_template(messages, tokenize, add_generation_prompt):
            if len(messages) == 1:
                return "<other-prefix>"
            return original(messages, tokenize, add_generation_prompt).replace("Answer", "ANSWER")

        tokenizer.apply_chat_template = bad_template

        with self.assertRaises(ValueError):
            _build_query_response_sft_row(
                tokenizer,
                prompt="Question",
                raw_response="Answer",
            )

    def test_build_query_response_sft_row_masks_template_scaffold(self):
        # Afterlife: only response content + end-of-turn marker carry loss; the
        # assistant marker, an empty think block, and trailing whitespace do not.
        from afterlife.trainer.qlora import _build_query_response_sft_row

        class QwenLikeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                text = f"<im_start>user\n{messages[0]['content']}<im_end>\n"
                if len(messages) > 1:
                    text += f"<im_start>assistant\n<think>\n\n</think>\n\n{messages[1]['content']}<im_end>\n"
                return text

            def __call__(self, text=None, **_kwargs):
                return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        row = _build_query_response_sft_row(QwenLikeTokenizer(), prompt="Q", raw_response="Answer")
        trained = "".join(chr(i) for i, m in zip(row["input_ids"], row["completion_mask"]) if m)
        self.assertEqual(trained, "Answer<im_end>")

    def test_build_query_response_sft_row_renders_multi_turn_context(self):
        # Afterlife: earlier turns are rendered by the chat template as real turns
        # (not folded into one user message) and carry no loss; only the final
        # response + end-of-turn marker does, even if its text appeared earlier.
        from afterlife.trainer.qlora import _build_query_response_sft_row

        class QwenLikeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                text = ""
                for i, m in enumerate(messages):
                    think = "<think>\n\n</think>\n\n" if m["role"] == "assistant" and i == len(messages) - 1 else ""
                    text += f"<im_start>{m['role']}\n{think}{m['content']}<im_end>\n"
                return text

            def __call__(self, text=None, **_kwargs):
                return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        context = [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "Same"},
            {"role": "user", "content": "Q2"},
        ]
        row = _build_query_response_sft_row(QwenLikeTokenizer(), raw_response="Same", context=context)
        text = "".join(chr(i) for i in row["input_ids"])
        self.assertIn("<im_start>system\nbe brief<im_end>", text)
        self.assertIn("<im_start>assistant\nSame<im_end>\n<im_start>user\nQ2", text)
        trained = "".join(chr(i) for i, m in zip(row["input_ids"], row["completion_mask"]) if m)
        self.assertEqual(trained, "Same<im_end>")

    def test_loader_reads_message_rows_and_prompt_rows(self):
        from afterlife.trainer.qlora import _load_self_generated_sft_rows

        class Tok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

            def __call__(self, text=None, **_kwargs):
                return {"input_ids": [ord(c) for c in text], "attention_mask": [1] * len(text)}

            def decode(self, ids):
                return "".join(chr(i) for i in ids)

        rows = [
            {"success": True, "prompt": "Q", "raw_response": "A"},
            {"success": True, "messages": [{"role": "user", "content": "Q1"},
                                           {"role": "assistant", "content": "A1"},
                                           {"role": "user", "content": "Q2"}], "raw_response": "A2"},
            {"success": True, "messages": [{"role": "user"}], "raw_response": "bad"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            loaded = _load_self_generated_sft_rows([path], Tok())
        self.assertEqual(len(loaded), 2)
        texts = ["".join(chr(i) for i in r["input_ids"]) for r in loaded]
        self.assertEqual(texts[1], "<user>Q1</user><assistant>A1</assistant><user>Q2</user><assistant>A2</assistant>")

    def test_build_query_response_sft_row_handles_thinking_channel_scaffold(self):
        # Regression: Gemma 4's chat template appends a thought-channel scaffold when
        # add_generation_prompt=True that is absent from the assistant-message
        # rendering, so the generation prompt is NOT a token-prefix of the full
        # conversation. _build_query_response_sft_row must render the boundary with
        # add_generation_prompt=False (assistant style) so masking still works.
        from afterlife.trainer.qlora import _build_query_response_sft_row

        class ThinkingTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                user = messages[0]["content"]
                base = f"<user>{user}</user><model>"
                if len(messages) == 1:
                    # add_generation_prompt=True injects a scaffold that diverges
                    # from the response; =False is the clean assistant-style prefix.
                    return base + ("<think>" if add_generation_prompt else "")
                return base + messages[1]["content"] + "</model>"

            def __call__(self, text=None, **_kwargs):
                assert isinstance(text, str)
                return {
                    "input_ids": [ord(c) for c in text],
                    "attention_mask": [1] * len(text),
                }

        row = _build_query_response_sft_row(
            ThinkingTokenizer(), prompt="Q", raw_response="Answer"
        )
        prefix = "<user>Q</user><model>"  # add_generation_prompt=False rendering
        full = "<user>Q</user><model>Answer</model>"
        self.assertEqual(row["input_ids"], [ord(c) for c in full])
        self.assertEqual(row["completion_mask"][: len(prefix)], [0] * len(prefix))
        self.assertEqual(
            row["completion_mask"][len(prefix) :],
            [1] * (len(full) - len(prefix)),
        )

    def test_completion_only_labels_mask_query_tokens(self):
        from afterlife.trainer.qlora import IGNORE_INDEX, _build_completion_only_labels

        labels = _build_completion_only_labels(
            input_ids=[10, 11, 12, 13],
            completion_mask=[0, 0, 1, 1],
        )

        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, 12, 13])

    def test_load_sft_dataset_multiple_files(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path0 = Path(tmpdir) / "gen0" / "sft_training_rows.jsonl"
            path1 = Path(tmpdir) / "gen1" / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(generation_id=0)], path0)
            self._make_jsonl([self._make_row(generation_id=1)], path1)
            dataset = _load_sft_dataset([path0, path1], tokenizer)
        self.assertEqual(len(dataset), 2)

    def test_load_sft_dataset_empty_input(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([], path)
            with self.assertRaises(ValueError, msg="No training rows found"):
                _load_sft_dataset([path], tokenizer)

    def test_load_sft_dataset_filters_unsuccessful(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(success=True), self._make_row(success=False)], path
            )
            dataset = _load_sft_dataset([path], tokenizer)
        self.assertEqual(len(dataset), 1)

    def test_load_sft_dataset_expands_virtual_epochs_with_rehearsal(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl(
                [self._make_row(prompt="Example task", raw_response="Answer")],
                path,
            )
            rehearsal = [
                {
                    "text": (
                        f"Reference rehearsal passage {i}. "
                        "Several independent systems interact in this example. "
                        "This sentence provides enough continuation text."
                    )
                }
                for i in range(8)
            ]

            with patch(
                "afterlife.trainer.qlora.load_dataset", return_value=rehearsal
            ) as mock_load:
                dataset = _load_sft_dataset(
                    [path],
                    tokenizer,
                    virtual_epochs=2,
                    rehearsal_dataset="example/rehearsal-corpus",
                    rehearsal_split="train",
                    rehearsal_text_field="text",
                    rehearsal_rows_per_epoch=2,
                    rehearsal_seed=11,
                    rehearsal_prompt="Continue this reference passage",
                    rehearsal_prompt_chars=32,
                    rehearsal_max_chars=160,
                )

        mock_load.assert_called_once_with(
            "example/rehearsal-corpus",
            split="train",
        )
        self.assertEqual(len(dataset), 6)
        texts = [
            "".join(chr(token_id) for token_id in row["input_ids"]) for row in dataset
        ]
        self.assertEqual(
            sum(
                text == "<user>Example task</user><asst>Answer</asst>" for text in texts
            ),
            2,
        )
        rehearsal_texts = [text for text in texts if "Continue this reference" in text]
        self.assertEqual(len(rehearsal_texts), 4)
        self.assertTrue(
            all("Reference rehearsal passage" in text for text in rehearsal_texts)
        )
        self.assertTrue(all(1 in row["completion_mask"] for row in dataset))

    def test_rehearsal_target_is_the_continuation_not_the_excerpt(self):
        from afterlife.trainer.qlora import _load_sft_dataset

        tokenizer = self._make_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(prompt="hi", raw_response="ok")], path)
            with patch("afterlife.trainer.qlora.load_dataset",
                       return_value=[{"text": "FIRSTPART then SECONDPART"}]):
                dataset = _load_sft_dataset(
                    [path], tokenizer, rehearsal_dataset="demo/rehearsal",
                    rehearsal_rows_per_epoch=1, rehearsal_prompt_chars=9,
                )
        row = next(r for r in dataset if "Continue" in "".join(map(chr, r["input_ids"])))
        target = "".join(chr(t) for t, m in zip(row["input_ids"], row["completion_mask"]) if m)
        self.assertIn("then SECONDPART", target)
        self.assertNotIn("FIRSTPART", target)

    def test_train_sft_rejects_weight_normalization_without_dft_weights(self):
        from afterlife.trainer.qlora import train_sft

        with self.assertRaisesRegex(ValueError, "needs DFT weights"):
            train_sft(base_model="m", training_data_paths=[], output_dir="out",
                      inner_loss="asft", asft_mode="sft+kl", asft_normalize_by="weights")

    @patch("afterlife.trainer.qlora.logger.warning")
    def test_warn_if_dataset_would_truncate_suggests_preserving_length(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from afterlife.trainer.qlora import _warn_if_dataset_would_truncate

        dataset = [
            {"input_ids": [1, 2]},
            {"input_ids": [1, 2, 3, 4, 5, 6]},
        ]

        _warn_if_dataset_would_truncate(dataset, MagicMock(), max_seq_length=4)

        mock_warning.assert_called_once()
        warning = mock_warning.call_args.args[0]
        self.assertIn("would truncate", warning)
        self.assertIn("max_seq_length=4", warning)
        self.assertIn("at least 6", warning)

    @patch("afterlife.trainer.qlora.logger.warning")
    def test_warn_if_dataset_would_truncate_stays_quiet_when_preserved(
        self,
        mock_warning: MagicMock,
    ) -> None:
        from afterlife.trainer.qlora import _warn_if_dataset_would_truncate

        dataset = [{"input_ids": [1, 2, 3]}]

        _warn_if_dataset_would_truncate(dataset, MagicMock(), max_seq_length=3)

        mock_warning.assert_not_called()


class TestQloraGpuCleanup(unittest.TestCase):
    @patch("afterlife.trainer.qlora.FastLanguageModel.from_pretrained")
    @patch("afterlife.trainer.qlora.gc.collect")
    def test_load_base_model_clears_torch_cuda_state_before_loading(
        self,
        mock_collect: MagicMock,
        mock_from_pretrained: MagicMock,
    ) -> None:
        from afterlife.trainer.qlora import _load_base_model

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        expected_model = MagicMock()
        expected_tokenizer = MagicMock()
        mock_from_pretrained.return_value = (expected_model, expected_tokenizer)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            model, tokenizer = _load_base_model("Qwen/Qwen2.5-Coder-7B-Instruct")

        mock_collect.assert_called_once()
        fake_torch.cuda.empty_cache.assert_called_once()
        self.assertIs(model, expected_model)
        self.assertIs(tokenizer, expected_tokenizer)

    @patch("afterlife.trainer.qlora.FastLanguageModel.from_pretrained")
    @patch("afterlife.trainer.qlora.gc.collect")
    def test_cleanup_torch_cuda_state_skips_when_cuda_unavailable(
        self,
        mock_collect: MagicMock,
        mock_from_pretrained: MagicMock,
    ) -> None:
        from afterlife.trainer.qlora import _load_base_model

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        mock_from_pretrained.return_value = (MagicMock(), MagicMock())

        with patch.dict(sys.modules, {"torch": fake_torch}):
            _load_base_model("Qwen/Qwen2.5-Coder-7B-Instruct")

        mock_collect.assert_called_once()
        fake_torch.cuda.empty_cache.assert_not_called()


class TestTrainSft(unittest.TestCase):
    def _make_jsonl(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def _make_row(self) -> dict:
        return {
            "prompt": "Do something",
            "raw_response": "Done",
            "timestamp": "2026-04-06T00:00:00",
            "interaction_type": "orchestrator",
            "success": True,
            "error_message": None,
            "episode_id": "ep_gen0_0001_123",
            "generation_id": 0,
            "episode_score": 100.0,
        }

    def _make_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2

        def apply_chat_template(messages, tokenize, add_generation_prompt):
            user_content = messages[0]["content"]
            if len(messages) == 1:
                return f"<user>{user_content}</user><asst>"
            asst_content = messages[1]["content"]
            return f"<user>{user_content}</user><asst>{asst_content}</asst>"

        def tokenize(text=None, **_kwargs):
            assert isinstance(text, str)
            return {
                "input_ids": [ord(char) for char in text],
                "attention_mask": [1] * len(text),
            }

        tokenizer.apply_chat_template.side_effect = apply_chat_template
        tokenizer.side_effect = tokenize
        return tokenizer

    def _run_train_sft(self, train_sft, **kwargs):
        with patch.dict(sys.modules, {"unsloth": MagicMock()}):
            return train_sft(**kwargs)

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_does_not_patch_dataset_map_for_preprocessed_rows(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer import qlora
        from afterlife.trainer.qlora import train_sft

        class FakeArrowDataset:
            def map(self, *args, **kwargs):
                return self

        original_map = FakeArrowDataset.map
        fake_arrow = SimpleNamespace(Dataset=FakeArrowDataset)
        fake_datasets = SimpleNamespace(arrow_dataset=fake_arrow)

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            with patch.dict(
                sys.modules,
                {"datasets": fake_datasets, "datasets.arrow_dataset": fake_arrow},
            ):
                with patch(
                    "afterlife.trainer.qlora._dataset_from_list",
                    side_effect=qlora._SimpleDataset.from_list,
                ):
                    self._run_train_sft(
                        train_sft,
                        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                        training_data_paths=[str(data_path)],
                        output_dir=str(output_dir),
                        max_steps=1,
                    )

        self.assertIs(FakeArrowDataset.map, original_map)
        self.assertFalse(hasattr(FakeArrowDataset.map, "_nsl_inproc"))

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._write_metadata")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_rejects_checkpointing_for_non_lora_exports(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_write_metadata,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_path = Path(tmpdir) / "artifact.gguf"
            mock_save_training_artifact.return_value = output_path.resolve()

            cases = [
                {"export_format": "gguf", "save_steps": 40},
                {"export_format": "merged_16bit", "resume_from_checkpoint": True},
            ]
            for case in cases:
                with self.subTest(case=case):
                    with self.assertRaisesRegex(
                        ValueError,
                        "checkpointing is only supported for LoRA",
                    ):
                        self._run_train_sft(
                            train_sft,
                            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                            training_data_paths=[str(data_path)],
                            output_dir=str(output_path),
                            max_steps=1,
                            **case,
                        )

        mock_load_base_model.assert_not_called()
        mock_attach_lora_adapter.assert_not_called()
        mock_sft_trainer_cls.assert_not_called()
        mock_save_training_artifact.assert_not_called()
        mock_write_metadata.assert_not_called()

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_runs_trainer(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
            )

        trainer_instance.train.assert_called_once()
        trainer_args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertTrue(trainer_args.completion_only_loss)
        self.assertEqual(trainer_args.dataset_kwargs, {"skip_prepare_dataset": True})
        self.assertFalse(trainer_args.remove_unused_columns)
        trainer_dataset = mock_sft_trainer_cls.call_args.kwargs["train_dataset"]
        self.assertIn("input_ids", trainer_dataset.column_names)
        self.assertIn("completion_mask", trainer_dataset.column_names)
        self.assertIn("data_collator", mock_sft_trainer_cls.call_args.kwargs)
        model.save_pretrained.assert_called_once()
        tokenizer.save_pretrained.assert_called_once()
        model.save_pretrained_merged.assert_not_called()
        self.assertEqual(result, output_dir.resolve())

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_loads_wandb_api_key_from_dotenv(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            env_path = Path(tmpdir) / ".env"
            env_path.write_text('WANDB_API_KEY="dotenv-key"\n', encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                with patch("afterlife.trainer.qlora.find_dotenv", return_value=str(env_path)):
                    result = self._run_train_sft(
                        train_sft,
                        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                        training_data_paths=[str(data_path)],
                        output_dir=str(output_dir),
                        max_steps=1,
                        wandb_project="demo-project",
                    )

                self.assertEqual("dotenv-key", os.environ["WANDB_API_KEY"])
                self.assertEqual("demo-project", os.environ["WANDB_PROJECT"])

        trainer_instance.train.assert_called_once()
        self.assertEqual(result, output_dir.resolve())

    def test_train_sft_wandb_project_still_errors_without_env_or_dotenv(self):
        from afterlife.trainer.qlora import train_sft

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            with patch.dict(os.environ, {}, clear=True):
                with patch("afterlife.trainer.qlora.find_dotenv", return_value=""):
                    with self.assertRaises(RuntimeError):
                        self._run_train_sft(
                            train_sft,
                            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                            training_data_paths=[str(data_path)],
                            output_dir=str(output_dir),
                            max_steps=1,
                            wandb_project="demo-project",
                        )

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_merged_calls_save_pretrained_merged(
        self, mock_load_base_model, mock_attach_lora_adapter, mock_sft_trainer_cls
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model

        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"

            result = self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                export_format="merged_16bit",
            )

        trainer_instance.train.assert_called_once()
        model.save_pretrained_merged.assert_called_once()
        model.save_pretrained.assert_not_called()
        self.assertEqual(result, output_dir.resolve())

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_passes_export_format_to_artifact_saver(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "artifact"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                export_format="merged_16bit",
            )

        mock_save_training_artifact.assert_called_once()
        self.assertEqual(
            mock_save_training_artifact.call_args.kwargs["export_format"],
            "merged_16bit",
        )

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    @patch("afterlife.trainer.qlora._install_fused_dft")
    def test_train_sft_uses_fused_dft_by_default(
        self,
        mock_install_fused_dft,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("UNSLOTH_ENABLE_CCE", None)
                self._run_train_sft(
                    train_sft,
                    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                    training_data_paths=[str(data_path)],
                    output_dir=str(output_dir),
                    max_steps=1,
                    inner_loss="dft",
                )
                self.assertEqual(os.environ["UNSLOTH_ENABLE_CCE"], "0")

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(args.loss_type, "nll")
        self.assertEqual(metadata["inner_loss"], "dft")
        self.assertEqual(metadata["dft_impl"], "fused")
        mock_install_fused_dft.assert_called_once()

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_keeps_trl_dft_impl_selectable(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer import qlora
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            with patch(
                "afterlife.trainer.qlora._dataset_from_list",
                side_effect=qlora._SimpleDataset.from_list,
            ):
                self._run_train_sft(
                    train_sft,
                    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                    training_data_paths=[str(data_path)],
                    output_dir=str(output_dir),
                    max_steps=1,
                    inner_loss="dft",
                    dft_impl="trl",
                )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(args.loss_type, "dft")
        self.assertEqual(metadata["inner_loss"], "dft")
        self.assertEqual(metadata["dft_impl"], "trl")

    @patch("afterlife.trainer.qlora._load_asft_classes")
    @patch("afterlife.trainer.qlora._install_fused_dft")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_asft_uses_asft_trainer_and_metadata(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_install_fused_dft,
        mock_load_asft_classes,
    ):
        from afterlife.trainer import qlora
        from afterlife.trainer.qlora import train_sft

        class FakeStreamingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeASFTTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.train = MagicMock()

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        trainer_instance = FakeASFTTrainer()
        fake_trainer_cls = MagicMock(return_value=trainer_instance)
        fake_trainer_cls.__module__ = "afterlife.trainer.asft_trainer"
        fake_trainer_cls.__qualname__ = "ASFTTrainer"
        mock_load_asft_classes.return_value = (
            qlora._SimpleSFTConfig,
            fake_trainer_cls,
            FakeStreamingConfig,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                inner_loss="asft",
                asft_mode="asft",
                asft_kl_weight=0.25,
                asft_kl_direction="reverse",
                asft_reference_policy="disable_adapter",
                asft_streaming="batch",
                asft_ref_microbatch_size=2,
                asft_seq_chunk_size=128,
                asft_normalize_by="weights",
            )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        args = fake_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(args.loss_type, "nll")
        self.assertEqual(fake_trainer_cls.call_args.kwargs["asft_mode"], "asft")
        self.assertEqual(fake_trainer_cls.call_args.kwargs["kl_weight"], 0.25)
        self.assertEqual(fake_trainer_cls.call_args.kwargs["kl_direction"], "reverse")
        self.assertEqual(
            fake_trainer_cls.call_args.kwargs["reference_policy"],
            "disable_adapter",
        )
        self.assertEqual(fake_trainer_cls.call_args.kwargs["normalize_by"], "weights")
        streaming_config = fake_trainer_cls.call_args.kwargs["asft_streaming"]
        self.assertEqual(
            streaming_config.kwargs,
            {
                "mode": "batch",
                "enabled": True,
                "ref_microbatch_size": 2,
                "seq_chunk_size": 128,
            },
        )
        self.assertEqual(metadata["inner_loss"], "asft")
        self.assertIsNone(metadata["dft_impl"])
        self.assertEqual(metadata["asft_mode"], "asft")
        self.assertEqual(metadata["asft_kl_weight"], 0.25)
        self.assertEqual(metadata["asft_kl_direction"], "reverse")
        self.assertEqual(metadata["asft_reference_policy"], "disable_adapter")
        self.assertEqual(metadata["asft_streaming"], "batch")
        self.assertEqual(metadata["asft_ref_microbatch_size"], 2)
        self.assertEqual(metadata["asft_seq_chunk_size"], 128)
        self.assertEqual(metadata["asft_normalize_by"], "weights")
        self.assertEqual(
            metadata["trainer_class"], "afterlife.trainer.asft_trainer.ASFTTrainer"
        )
        mock_install_fused_dft.assert_not_called()

    def test_load_asft_classes_errors_without_vendored_modules(self):
        from afterlife.trainer import qlora
        from afterlife.trainer.qlora import _load_asft_classes

        # Force the local-module import to fail and ensure the cached globals do
        # not short-circuit the import attempt.
        with (
            patch.object(qlora, "ASFTTrainer", None),
            patch.object(qlora, "ASFTStreamingConfig", None),
            patch.dict(sys.modules, {"afterlife.trainer.asft_trainer": None}),
        ):
            with self.assertRaisesRegex(RuntimeError, "vendored ASFT modules"):
                _load_asft_classes()

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_writes_metadata(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row(), self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
                learning_rate=1e-4,
            )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        self.assertEqual(metadata["base_model"], "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(metadata["learning_rate"], 1e-4)
        self.assertEqual(metadata["row_count"], 2)
        self.assertIn("training_data_paths", metadata)
        self.assertIn("exported_at", metadata)
        self.assertIn("max_steps", metadata)
        self.assertEqual(metadata["inner_loss"], "sft")

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_uses_virtual_epoch_rehearsal_and_sft_knobs(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()
            rehearsal = [
                {
                    "text": (
                        f"Rehearsal passage {i}. "
                        "Additional context and process details continue here."
                    )
                }
                for i in range(8)
            ]

            with patch("afterlife.trainer.qlora.load_dataset", return_value=rehearsal):
                self._run_train_sft(
                    train_sft,
                    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                    training_data_paths=[str(data_path)],
                    output_dir=str(output_dir),
                    max_steps=-1,
                    num_train_epochs=2,
                    gradient_accumulation_steps=16,
                    learning_rate=1e-4,
                    warmup_ratio=0.03,
                    lr_scheduler_type="linear",
                    weight_decay=0.001,
                    lora_rank=32,
                    lora_alpha=32,
                    lora_dropout=0.0,
                    seed=3407,
                    rehearsal_dataset="example/rehearsal-corpus",
                    rehearsal_rows_per_epoch=2,
                    rehearsal_seed=123,
                )

            metadata = json.loads(
                (output_dir.resolve() / "training_info.json").read_text()
            )

        mock_attach_lora_adapter.assert_called_once()
        self.assertEqual(mock_attach_lora_adapter.call_args.args[1]["rank"], 32)
        self.assertEqual(mock_attach_lora_adapter.call_args.args[1]["alpha"], 32)
        trainer_dataset = mock_sft_trainer_cls.call_args.kwargs["train_dataset"]
        self.assertEqual(len(trainer_dataset), 6)
        trainer_args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(trainer_args.max_steps, -1)
        self.assertEqual(trainer_args.num_train_epochs, 1)
        self.assertEqual(trainer_args.gradient_accumulation_steps, 16)
        self.assertEqual(trainer_args.learning_rate, 1e-4)
        self.assertEqual(trainer_args.warmup_ratio, 0.03)
        self.assertEqual(trainer_args.lr_scheduler_type.value, "linear")
        self.assertEqual(trainer_args.weight_decay, 0.001)
        self.assertEqual(metadata["configured_num_train_epochs"], 2)
        self.assertEqual(metadata["virtual_epochs"], 2)
        self.assertEqual(metadata["row_count"], 6)
        self.assertEqual(metadata["rehearsal_rows_per_epoch"], 2)

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_checkpointing_enables_step_saves(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        trainer_instance = MagicMock()
        mock_sft_trainer_cls.return_value = trainer_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=-1,
                save_steps=40,
            )

        args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(args.save_strategy, "steps")
        self.assertEqual(args.save_steps, 40)
        self.assertGreaterEqual(args.save_total_limit, 1)
        # No --resume-from-checkpoint -> fresh run (None passed to train()).
        trainer_instance.train.assert_called_once_with(resume_from_checkpoint=None)

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_default_disables_checkpointing(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=1,
            )

        args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertEqual(args.save_strategy, "no")

    @patch("afterlife.trainer.qlora.SFTTrainer")
    @patch("afterlife.trainer.qlora._save_training_artifact")
    @patch("afterlife.trainer.qlora._attach_lora_adapter")
    @patch("afterlife.trainer.qlora._load_base_model")
    def test_train_sft_group_by_length_sets_sampler_flag(
        self,
        mock_load_base_model,
        mock_attach_lora_adapter,
        mock_save_training_artifact,
        mock_sft_trainer_cls,
    ):
        from afterlife.trainer.qlora import train_sft

        model = MagicMock()
        tokenizer = self._make_tokenizer()
        mock_load_base_model.return_value = (model, tokenizer)
        mock_attach_lora_adapter.return_value = model
        mock_sft_trainer_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "sft_training_rows.jsonl"
            self._make_jsonl([self._make_row()], data_path)
            output_dir = Path(tmpdir) / "adapter"
            mock_save_training_artifact.return_value = output_dir.resolve()

            self._run_train_sft(
                train_sft,
                base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
                training_data_paths=[str(data_path)],
                output_dir=str(output_dir),
                max_steps=-1,
                per_device_train_batch_size=4,
                group_by_length=True,
            )

        args = mock_sft_trainer_cls.call_args.kwargs["args"]
        self.assertTrue(args.group_by_length)
        self.assertEqual(args.per_device_train_batch_size, 4)

    def test_train_sft_cli_accepts_checkpoint_flags(self):
        from afterlife.trainer.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/sft.jsonl",
                "--output",
                "./adapter",
                "--save-steps",
                "40",
                "--resume-from-checkpoint",
                "--group-by-length",
                "--inner-loss",
                "dft",
            ]
        )
        self.assertEqual(args.save_steps, 40)
        self.assertTrue(args.resume_from_checkpoint)
        self.assertTrue(args.group_by_length)
        self.assertEqual(args.inner_loss, "dft")
        self.assertEqual(args.dft_impl, "fused")

    def test_train_sft_cli_accepts_trl_dft_impl(self):
        from afterlife.trainer.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/sft.jsonl",
                "--output",
                "./adapter",
                "--inner-loss",
                "dft",
                "--dft-impl",
                "trl",
            ]
        )
        self.assertEqual(args.inner_loss, "dft")
        self.assertEqual(args.dft_impl, "trl")

    def test_train_sft_cli_accepts_asft_flags(self):
        from afterlife.trainer.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/sft.jsonl",
                "--output",
                "./adapter",
                "--inner-loss",
                "asft",
                "--asft-mode",
                "sft+kl",
                "--asft-kl-weight",
                "0.2",
                "--asft-kl-direction",
                "reverse",
                "--asft-reference-policy",
                "frozen_copy",
                "--asft-streaming",
                "hybrid",
                "--asft-ref-microbatch-size",
                "3",
                "--asft-seq-chunk-size",
                "256",
                "--asft-normalize-by",
                "weights",
            ]
        )
        self.assertEqual(args.inner_loss, "asft")
        self.assertEqual(args.asft_mode, "sft+kl")
        self.assertEqual(args.asft_kl_weight, 0.2)
        self.assertEqual(args.asft_kl_direction, "reverse")
        self.assertEqual(args.asft_reference_policy, "frozen_copy")
        self.assertEqual(args.asft_streaming, "hybrid")
        self.assertEqual(args.asft_ref_microbatch_size, 3)
        self.assertEqual(args.asft_seq_chunk_size, 256)
        self.assertEqual(args.asft_normalize_by, "weights")

    def test_train_sft_cli_multiple_training_data(self):
        """CLI should accept multiple --training-data arguments."""
        from afterlife.trainer.qlora import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--base-model",
                "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--training-data",
                "./data/gen0/sft.jsonl",
                "--training-data",
                "./data/gen1/sft.jsonl",
                "--output",
                "./models/adapters/test",
            ]
        )
        self.assertEqual(len(args.training_data), 2)
        self.assertIn("./data/gen0/sft.jsonl", args.training_data)
        self.assertIn("./data/gen1/sft.jsonl", args.training_data)


class TestResolveTargetModules(unittest.TestCase):
    """Tests target-module resolution for Unsloth/PEFT."""

    @staticmethod
    def _fake_model(names):
        # torch's named_modules() yields ('', root) first; mimic that.
        pairs = [("", object())] + [(n, object()) for n in names]
        return SimpleNamespace(named_modules=lambda: list(pairs))

    @staticmethod
    def _lm_module_names():
        names = []
        for layer in range(2):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                names.append(f"model.layers.{layer}.self_attn.{proj}")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                names.append(f"model.layers.{layer}.mlp.{proj}")
        names += ["model.embed_tokens", "model.norm", "lm_head"]
        return names

    def test_list_spec_passes_through_unchanged(self):
        from afterlife.trainer.qlora import _resolve_target_modules

        model = self._fake_model(self._lm_module_names())
        spec = ["q_proj", "v_proj"]
        self.assertEqual(_resolve_target_modules(model, spec), spec)

    def test_lm_regex_resolves_to_leaf_names(self):
        from afterlife.trainer.qlora import (
            LORA_TARGET_MODULES_LM_ONLY_REGEX,
            _resolve_target_modules,
        )

        model = self._fake_model(self._lm_module_names())
        result = _resolve_target_modules(model, LORA_TARGET_MODULES_LM_ONLY_REGEX)
        self.assertEqual(
            sorted(result),
            sorted(
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            ),
        )
        self.assertTrue(all("." not in m for m in result))

    def test_vlm_regex_excludes_vision_tower_via_full_paths(self):
        from afterlife.trainer.qlora import (
            LORA_TARGET_MODULES_LM_ONLY_REGEX,
            _resolve_target_modules,
        )

        names = self._lm_module_names()
        # Add excluded modules that reuse language-model leaf names.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append(f"vision_tower.encoder.layers.0.self_attn.{proj}")
        model = self._fake_model(names)
        result = _resolve_target_modules(model, LORA_TARGET_MODULES_LM_ONLY_REGEX)
        self.assertTrue(all("." in m for m in result))
        self.assertFalse(any("vision_tower" in m for m in result))
        self.assertEqual(len([m for m in result if m.endswith("q_proj")]), 2)

    def test_regex_matching_nothing_raises(self):
        from afterlife.trainer.qlora import _resolve_target_modules

        model = self._fake_model(["model.embed_tokens", "lm_head"])
        with self.assertRaises(ValueError):
            _resolve_target_modules(model, r".*\.(q_proj|v_proj)$")


if __name__ == "__main__":
    unittest.main()
