"""Doxygen input filter for the C files of tiopt.

Our code is documented with plain // comments, which Doxygen ignores, so this filter turns them into Doxygen comments
before Doxygen parses each file (the files themselves are left unchanged). A block of // comments at the start of a
line becomes the documentation of the declaration directly below it, with any @TODO lines moved into a todo section,
and a // comment after a member or enum value in a header becomes the documentation of that member. Comments inside
functions, comments followed by a blank line, headings of sections, and commented-out code are left alone.
"""

import re
import sys

DOC_LINE = re.compile(r"^// (\S.*)$")
TODO_LINE = re.compile(r"^// @TODO - (.*)$")
CONTINUATION_LINE = re.compile(r"^//\s{2,}(\S.*)$")
TRAILING_COMMENT = re.compile(r"^(\s*(?:[^/\s].*?[;,]|[A-Z_][A-Z0-9_]*|#define\s.*?))\s*// (.*)$")


def is_declaration(line):
    """Check whether a line starts something which Doxygen can document."""
    return bool(line.strip()) and not line.startswith(("//", "#include", "#pragma", "#endif", "{", "}"))


def is_section(block, lines, start):
    """Check whether a comment is the heading of a section (e.g. // Trap data) rather than documentation."""
    if len(block) == 1 and len(block[0].split()) <= 5 and lines[start].startswith(("struct", "enum", "#")):
        return True

    # A comment above several macros describes all of them
    n_macros = 0
    for line in lines[start:]:
        if not line.startswith("#"):
            break
        n_macros += line.startswith("#define")

    return n_macros > 1


def looks_like_code(text):
    """Check whether the text of a comment is commented-out code rather than prose."""
    return text.rstrip().endswith((";", "{", "}")) or text.lstrip().startswith(("printf", "return "))


def filter_block(block):
    """Convert a block of comment lines above a declaration, or return None if it should be left alone."""
    description = []
    todos = []
    for line in block:
        todo = TODO_LINE.match(line)
        continuation = CONTINUATION_LINE.match(line)
        doc = DOC_LINE.match(line)
        if todo:
            todos.append("/// @todo " + todo.group(1))
        elif continuation and todos:
            todos.append("///   " + continuation.group(1))
        elif continuation and description and not looks_like_code(continuation.group(1)):
            description.append("///   " + continuation.group(1))
        elif doc and not looks_like_code(doc.group(1)):
            description.append("/// " + doc.group(1))
        else:
            return None

    # Doxygen ends a todo section at the next paragraph, so the description goes first
    if description:
        description[0] = description[0].replace("/// ", "/// @brief ", 1)

    return description + todos


def filter_source(lines, header):
    output = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("//"):
            j = i
            while j < len(lines) and lines[j].startswith("//"):
                j += 1

            block = lines[i:j]
            documents = j < len(lines) and is_declaration(lines[j]) and not is_section(block, lines, j)
            converted = filter_block(block) if documents else None
            output.extend(converted if converted is not None else block)
            i = j
            continue

        line = lines[i]
        if header:
            trailing = TRAILING_COMMENT.match(line)
            if trailing and "@" not in trailing.group(2):
                line = trailing.group(1) + " ///< " + trailing.group(2)

        output.append(line)
        i += 1

    return output


def main():
    filename = sys.argv[1]
    with open(filename, encoding="utf-8") as source:
        lines = source.read().split("\n")

    sys.stdout.write("\n".join(filter_source(lines, filename.endswith(".h"))))


if __name__ == "__main__":
    main()
