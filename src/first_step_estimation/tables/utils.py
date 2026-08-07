"""Shared helpers for writing LaTeX table files."""


def write_latex_table(latex_table, output_path):
    """
    Write a LaTeX tabular to disk in the form pre-commit hooks would leave it.

    Strips trailing whitespace from each line (trailing-whitespace hook) and
    ensures the file ends with exactly one newline (end-of-file-fixer hook),
    so the generated file matches what gets committed.
    """
    lines = [line.rstrip() for line in latex_table.split("\n")]
    cleaned = "\n".join(lines).rstrip("\n") + "\n"

    with open(output_path, "w") as f:
        f.write(cleaned)
