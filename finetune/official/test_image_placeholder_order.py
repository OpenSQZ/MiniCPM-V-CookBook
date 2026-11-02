import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


DATASET_PATH = Path(__file__).with_name("dataset.py")


class ImagePlaceholderOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "minicpm_dataset_image_order", DATASET_PATH
        )
        cls.dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.dataset)

    def setUp(self):
        self.tokenizer = SimpleNamespace(
            im_start="<im_start>",
            unk_token="<unk>",
            im_end="<im_end>",
            slice_start="<slice_start>",
            slice_end="<slice_end>",
            im_id_start="<image_id>",
            im_id_end="</image_id>",
        )
        self.images = {
            "<image_00>": Image.new("RGB", (28, 28), "red"),
            "<image_01>": Image.new("RGB", (28, 28), "blue"),
        }

    @staticmethod
    def transform(image):
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()

    @staticmethod
    def colors(result):
        return [
            tuple(image[:, 0, 0].tolist()) for image in result["pixel_values"]
        ]

    def preprocess(self, images, messages, llm_type="llama3", **kwargs):
        def capture_conversation(conversations, *args):
            return {"conversations": conversations}

        with patch.object(
            self.dataset, "conversation_to_ids", capture_conversation
        ):
            return self.dataset.preprocess(
                images,
                messages,
                self.tokenizer,
                self.transform,
                query_nums=2,
                llm_type=llm_type,
                **kwargs,
            )

    @staticmethod
    def messages(content):
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "An answer"},
        ]

    def test_features_follow_placeholder_order_and_preserve_messages(self):
        messages = self.messages("<image_01> then <image_00>")
        original = copy.deepcopy(messages)
        result = self.preprocess(self.images, messages)
        self.assertEqual(self.colors(result), [(0, 0, 255), (255, 0, 0)])
        self.assertEqual(messages, original)

    def test_repeated_placeholders_receive_features_in_each_turn(self):
        messages = self.messages("<image_01> <image_00>") + [
            {"role": "user", "content": "Look at <image_01> again"},
            {"role": "assistant", "content": "Blue"},
        ]
        result = self.preprocess(self.images, messages)
        self.assertEqual(
            self.colors(result), [(0, 0, 255), (255, 0, 0), (0, 0, 255)]
        )

    def test_unused_images_do_not_create_unmatched_features(self):
        transformed = []
        original_transform = self.transform

        def transform(image):
            transformed.append(image.getpixel((0, 0)))
            return original_transform(image)

        with patch.object(self, "transform", transform):
            result = self.preprocess(
                self.images, self.messages("<image_01>")
            )
        self.assertEqual(self.colors(result), [(0, 0, 255)])
        self.assertEqual(transformed, [(0, 0, 255)])

    def test_sliced_image_groups_are_reordered_together(self):
        self.images["<image_00>"] = Image.new("RGB", (84, 56), "red")
        config = {
            "patch_size": 14,
            "max_slice_nums": 4,
            "scale_resolution": 28,
        }
        result = self.preprocess(
            self.images,
            self.messages("<image_01> <image_00>"),
            slice_config=config,
        )
        _, patches, _ = self.dataset.slice_image(
            self.images["<image_00>"], 4, 28, 14
        )
        red_count = 1 + sum(len(row) for row in patches)
        self.assertGreater(red_count, 1)
        self.assertEqual(
            self.colors(result), [(0, 0, 255)] + [(255, 0, 0)] * red_count
        )

    def test_qwen_sliced_groups_keep_matching_image_ids(self):
        config = {
            "patch_size": 14,
            "max_slice_nums": 4,
            "scale_resolution": 28,
        }
        self.images = {
            "<image_00>": Image.new("RGB", (84, 56), "red"),
            "<image_01>": Image.new("RGB", (84, 56), "blue"),
        }
        result = self.preprocess(
            self.images,
            self.messages("<image_01> <image_00>"),
            llm_type="qwen",
            slice_config=config,
        )
        _, patches, _ = self.dataset.slice_image(
            self.images["<image_00>"], 4, 28, 14
        )
        group_size = 1 + sum(len(row) for row in patches)
        self.assertEqual(
            self.colors(result),
            [(0, 0, 255)] * group_size + [(255, 0, 0)] * group_size,
        )
        prompt = result["conversations"][0]["content"]
        self.assertLess(
            prompt.index("<image_id>1</image_id>"),
            prompt.index("<image_id>0</image_id>"),
        )

    def test_batch_vision_sizes_follow_reordered_images(self):
        self.images = {
            "<image_00>": Image.new("RGB", (42, 28), "red"),
            "<image_01>": Image.new("RGB", (28, 56), "blue"),
        }
        result = self.preprocess(
            self.images,
            self.messages("<image_01> <image_00>"),
            batch_vision=True,
        )
        self.assertEqual(result["tgt_sizes"].tolist(), [[4, 2], [2, 3]])
        self.assertEqual(self.colors(result), [(0, 0, 255), (255, 0, 0)])

    def test_in_order_inputs_keep_existing_behavior(self):
        result = self.preprocess(
            self.images, self.messages("<image_00> <image_01>")
        )
        self.assertEqual(self.colors(result), [(255, 0, 0), (0, 0, 255)])

    def test_single_image_implicit_prefix_keeps_existing_behavior(self):
        result = self.preprocess(
            {"<image>": self.images["<image_00>"]},
            self.messages("What color?"),
        )
        self.assertEqual(self.colors(result), [(255, 0, 0)])
        self.assertTrue(
            result["conversations"][0]["content"].startswith(
                "<im_start><unk><unk><im_end>"
            )
        )


if __name__ == "__main__":
    unittest.main()
