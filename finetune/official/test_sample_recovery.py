import importlib.util
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image


DATASET_PATH = Path(__file__).with_name("dataset.py")


class SampleRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "minicpm_dataset_recovery", DATASET_PATH
        )
        cls.dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dataset)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calls = []

    def row(self, name, color=None, conversations=None):
        path = self.root / name
        if color is not None:
            Image.new("RGB", (2, 2), color).save(path)
        return {
            "image": str(path),
            "conversations": [] if conversations is None else conversations,
        }

    def make_dataset(self, rows):
        return self.dataset.SupervisedDataset(
            rows, transform=None, tokenizer=None, slice_config=None
        )

    def preprocess(self, images, conversations, *args, **kwargs):
        value = images["<image>"].getpixel((0, 0))[0]
        self.calls.append(value)
        return {
            "input_ids": torch.tensor([1, value]),
            "position_ids": torch.tensor([0, 1]),
            "target": torch.tensor([-100, value]),
            "pixel_values": [],
            "tgt_sizes": [],
            "image_bound": [],
        }

    def test_corrupt_record_recovers_without_invalid_random_index(self):
        rows = [
            self.row("missing.png"),
            self.row("valid.png", (12, 34, 56)),
        ]
        draws = []

        def choose_upper(low, high):
            draws.append((low, high))
            if len(draws) > 2:
                raise RuntimeError("Unbounded invalid-index recovery")
            return high

        with patch.object(self.dataset, "preprocess", self.preprocess), patch.object(
            random, "randint", choose_upper
        ):
            result = self.make_dataset(rows)[0]
        self.assertEqual(result["input_ids"].tolist(), [1, 12])
        self.assertEqual(result["attention_mask"].tolist(), [True, True])

    def test_all_invalid_records_are_tried_once_with_original_cause(self):
        rows = [self.row(f"missing-{index}.png") for index in range(3)]
        attempts = []
        original_open = self.dataset.Image.open
        draws = []

        def track_open(path):
            attempts.append(path)
            return original_open(path)

        def bounded_legacy_draw(low, high):
            draws.append((low, high))
            if len(draws) > len(rows):
                raise RuntimeError("Unbounded legacy recovery")
            return 0

        with patch.object(self.dataset.Image, "open", track_open), patch.object(
            random, "randint", bounded_legacy_draw
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Unable to load any of 3 training samples"
            ) as error:
                self.make_dataset(rows)[2]

        self.assertIsInstance(error.exception.__cause__, FileNotFoundError)
        self.assertEqual(
            attempts, [rows[index]["image"] for index in (2, 0, 1)]
        )

    def test_requested_valid_record_is_not_replaced(self):
        rows = [
            self.row("first.png", (12, 34, 56)),
            self.row("second.png", (78, 90, 12)),
        ]
        with patch.object(self.dataset, "preprocess", self.preprocess):
            result = self.make_dataset(rows)[1]
        self.assertEqual(result["labels"].tolist(), [-100, 78])
        self.assertEqual(self.calls, [78])

    def test_negative_index_keeps_sequence_semantics(self):
        rows = [
            self.row("first.png", (12, 34, 56)),
            self.row("second.png", (78, 90, 12)),
        ]
        with patch.object(self.dataset, "preprocess", self.preprocess):
            result = self.make_dataset(rows)[-1]
        self.assertEqual(result["input_ids"].tolist(), [1, 78])

    def test_recovery_wraps_to_the_first_record(self):
        rows = [
            self.row("valid.png", (12, 34, 56)),
            self.row("missing.png"),
        ]
        with patch.object(self.dataset, "preprocess", self.preprocess), patch.object(
            random, "randint", return_value=0
        ):
            result = self.make_dataset(rows)[1]
        self.assertEqual(result["input_ids"].tolist(), [1, 12])

    def test_external_out_of_range_indices_are_not_resampled(self):
        rows = [self.row("valid.png", (12, 34, 56))]
        with patch.object(
            random, "randint", side_effect=AssertionError("Unexpected resampling")
        ):
            for index in (-2, 1, 2):
                with self.subTest(index=index), self.assertRaises(IndexError):
                    self.make_dataset(rows)[index]

    def test_empty_dataset_reports_index_error(self):
        with patch.object(
            random, "randint", side_effect=AssertionError("Unexpected resampling")
        ):
            with self.assertRaises(IndexError):
                self.make_dataset([])[0]

    def test_process_control_exceptions_are_not_swallowed(self):
        rows = [self.row("valid.png", (12, 34, 56))]
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception.__name__):
                with patch.object(
                    self.dataset, "preprocess", side_effect=exception
                ), patch.object(
                    random, "randint", side_effect=AssertionError("Unexpected resampling")
                ):
                    with self.assertRaises(exception):
                        self.make_dataset(rows)[0]

    def test_preprocessing_error_can_recover_to_a_valid_record(self):
        rows = [
            self.row("bad.png", (12, 34, 56), conversations="invalid"),
            self.row("valid.png", (78, 90, 12)),
        ]

        def preprocess(images, conversations, *args, **kwargs):
            if conversations == "invalid":
                raise ValueError("Invalid conversation")
            return self.preprocess(images, conversations, *args, **kwargs)

        with patch.object(self.dataset, "preprocess", preprocess), patch.object(
            random, "randint", return_value=1
        ):
            result = self.make_dataset(rows)[0]
        self.assertEqual(result["input_ids"].tolist(), [1, 78])


if __name__ == "__main__":
    unittest.main()
