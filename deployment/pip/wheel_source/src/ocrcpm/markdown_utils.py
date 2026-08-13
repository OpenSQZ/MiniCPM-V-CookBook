from collections import Counter
import re


DISPLAY_BRACKET_SEP_RE = re.compile(re.escape(r"\]") + r"\s*" + re.escape(r"\["))


def truncate_repetitive_content(content: str, min_count: int = 50) -> str:
    if not content or len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content

    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if len(lines) >= 10:
        most_common, count = Counter(lines).most_common(1)[0]
        if count >= 10 and count / len(lines) >= 0.8:
            return most_common
    return content


def postprocess_content(content: str, output_format: str, label: str = "") -> str:
    if content is None:
        return ""
    content = truncate_repetitive_content(content.strip(), min_count=5000 if label == "table" else 50)

    if output_format == "latex":
        for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
            if content.startswith(left) and content.endswith(right):
                content = content[len(left): len(content) - len(right)].strip()

        # Normalize noisy display-formula separators like "\] \[".
        if label == "display_formula":
            content = DISPLAY_BRACKET_SEP_RE.sub("$$   $$", content)
            content = content.replace("\\[", "").replace("\\]", "")
    elif output_format == "html":
        if content.startswith("```html"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        # Guard against truncated long-table generations. Missing </table>
        # can force TEDS to zero even when most rows are correct.
        if label == "table":
            open_count = content.count("<table")
            close_count = content.count("</table>")
            if open_count > close_count:
                content = content + ("</table>" * (open_count - close_count))
    return content


def assemble_markdown(blocks: list, prompt_cfg: dict) -> str:
    assembly = prompt_cfg.get("markdown_assembly", {})
    ignore_labels = set(assembly.get("ignore_labels", []))
    title_fmt = assembly.get("title_format", {})
    formula_fmt = assembly.get("formula_format", {})
    table_fmt = assembly.get("table_format", "{content}")
    image_fmt = assembly.get("image_format", "![image]()")
    chart_fmt = assembly.get("chart_format", "{content}")
    seal_fmt = assembly.get("seal_format", "{content}")

    parts = []
    for block in sorted(blocks, key=lambda x: x.get("order", 0)):
        label = block.get("label", "")
        content = block.get("content", "")
        if label in ignore_labels:
            continue
        if not content and label not in ("image", "header_image", "footer_image", "chart", "seal"):
            continue
        if label in title_fmt:
            parts.append(title_fmt[label].format(content=content))
        elif label in formula_fmt:
            parts.append(formula_fmt[label].format(content=content))
        elif label == "table":
            parts.append(table_fmt.format(content=content))
        elif label in ("image", "header_image", "footer_image"):
            parts.append(image_fmt)
        elif label == "chart":
            parts.append(chart_fmt.format(content=content) if content else image_fmt)
        elif label == "seal":
            parts.append(seal_fmt.format(content=content) if content else image_fmt)
        else:
            parts.append(content)
    return "\n\n".join(parts)
