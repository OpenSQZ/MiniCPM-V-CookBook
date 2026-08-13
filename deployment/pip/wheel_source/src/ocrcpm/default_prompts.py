DEFAULT_PROMPT_MAP = {
    "default": {
        "prompt": "Text Recognition:",
        "max_new_tokens": 8192,
        "output_format": "text",
        "resize_min_side": None,
        "slice_nums": 9,
    },
    "element_prompts": {
        "text": {"prompt": "Text Recognition:", "max_new_tokens": 4096, "output_format": "text"},
        "content": {"prompt": "Text Recognition:", "max_new_tokens": 4096, "output_format": "text"},
        "abstract": {"prompt": "Text Recognition:", "max_new_tokens": 4096, "output_format": "text"},
        "reference": {"prompt": "Text Recognition:", "max_new_tokens": 2048, "output_format": "text"},
        "reference_content": {"prompt": "Text Recognition:", "max_new_tokens": 2048, "output_format": "text"},
        "vertical_text": {"prompt": "Text Recognition:", "max_new_tokens": 2048, "output_format": "text"},
        "vision_footnote": {"prompt": "Text Recognition:", "max_new_tokens": 1024, "output_format": "text"},
        "algorithm": {"prompt": "Text Recognition:", "max_new_tokens": 4096, "output_format": "text"},
        "doc_title": {"prompt": "Text Recognition:", "max_new_tokens": 512, "output_format": "text"},
        "paragraph_title": {"prompt": "Text Recognition:", "max_new_tokens": 512, "output_format": "text"},
        "figure_title": {"prompt": "Text Recognition:", "max_new_tokens": 512, "output_format": "text"},
        "table": {"prompt": "Table Recognition:", "max_new_tokens": 8192, "output_format": "html"},
        # Align with eval framework display_formula_v3 mode:
        # display formula uses Formula Recognition, inline formula keeps Text Recognition.
        "display_formula": {"prompt": "Formula Recognition:", "max_new_tokens": 2048, "output_format": "latex"},
        "inline_formula": {"prompt": "Text Recognition:", "max_new_tokens": 1024, "output_format": "latex"},
        "formula_number": {"prompt": "Text Recognition:", "max_new_tokens": 256, "output_format": "text"},
        "header": {"prompt": "Text Recognition:", "max_new_tokens": 512, "output_format": "text"},
        "footer": {"prompt": "Text Recognition:", "max_new_tokens": 512, "output_format": "text"},
        "footnote": {"prompt": "Text Recognition:", "max_new_tokens": 1024, "output_format": "text"},
        "aside_text": {"prompt": "Text Recognition:", "max_new_tokens": 1024, "output_format": "text"},
        "number": {"prompt": "Text Recognition:", "max_new_tokens": 256, "output_format": "text"},
        "image": {"skip": True},
        "header_image": {"skip": True},
        "footer_image": {"skip": True},
        "chart": {"skip": True},
        "seal": {"skip": True},
    },
    "markdown_assembly": {
        "ignore_labels": ["number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text"],
        "title_format": {"doc_title": "# {content}", "paragraph_title": "## {content}", "figure_title": "*{content}*"},
        "formula_format": {"display_formula": "$$\n{content}\n$$", "inline_formula": "${content}$"},
        "table_format": "{content}",
        "image_format": "![image]()",
        "chart_format": "{content}",
        "seal_format": "{content}",
    },
}


def apply_prompt_override_mode(prompt_map: dict, mode: str) -> dict:
    normalized = (mode or "legacy").strip().lower()
    if normalized in ("legacy", "none"):
        return prompt_map
    if normalized in ("display_formula_v3", "formula_recognition"):
        elems = prompt_map.setdefault("element_prompts", {})
        elems.setdefault("display_formula", {})["prompt"] = "Formula Recognition:"
        return prompt_map
    raise ValueError(f"Unknown prompt override mode: {mode}")
