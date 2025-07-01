def inject_latex_table_note(
    latex_code: str, note_text: str, parbox_width="1\\textwidth"
) -> str:
    """
    Injects a note into a LaTeX table after the tabular environment.

    Parameters:
    - latex_code (str): The original LaTeX code of the table.
    - note_text (str): The note to insert (plain text, no need to wrap in LaTeX).
    - parbox_width (str): Width of the parbox (default: '1\\textwidth').

    Returns:
    - str: Modified LaTeX code with the note inserted.
    """
    note_latex = (
        f"\n\\vspace{{0.5em}}\n"
        f"\\footnotesize{{\\parbox{{{parbox_width}}}{{\\textbf{{Note}}: {note_text}}}}}\n"
    )

    insert_pos = latex_code.find(r"\end{tabular}")
    if insert_pos == -1:
        raise ValueError("Could not find \\end{tabular} in the LaTeX code.")

    modified_latex = (
        latex_code[: insert_pos + len(r"\end{tabular}")]
        + note_latex
        + latex_code[insert_pos + len(r"\end{tabular}") :]
    )

    return modified_latex
