import ast
import io


def add_line_numbers(code: str) -> str:
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(
        f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines)
    )


def line_range(node: ast.AST) -> tuple[int, int] | None:
    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
        return (node.lineno, node.end_lineno)
    return None


def extract_line_numbers(code: str) -> tuple[list[str], int]:
    lines = code.split("\n")
    return lines, len(lines)
