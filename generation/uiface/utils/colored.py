# https://github.com/hfeeki/termcolor/blob/master/termcolor.py

import os
import re

ATTRIBUTES = dict(
    list(
        zip(
            ["bold", "dark", "", "underline", "blink", "", "reverse", "concealed"],
            list(range(1, 9)),
        )
    )
)
del ATTRIBUTES[""]

ATTRIBUTES_RE = "\033\\[(?:%s)m" % "|".join(["%d" % v for v in ATTRIBUTES.values()])

HIGHLIGHTS = dict(
    list(
        zip(
            [
                "on_grey",
                "on_red",
                "on_green",
                "on_yellow",
                "on_blue",
                "on_magenta",
                "on_cyan",
                "on_white",
            ],
            list(range(40, 48)),
        )
    )
)

HIGHLIGHTS_RE = "\033\\[(?:%s)m" % "|".join(["%d" % v for v in HIGHLIGHTS.values()])

COLORS = dict(
    list(
        zip(
            [
                "grey",
                "red",
                "green",
                "yellow",
                "blue",
                "magenta",
                "cyan",
                "white",
            ],
            list(range(30, 38)),
        )
    )
)

COLORS_RE = "\033\\[(?:%s)m" % "|".join(["%d" % v for v in COLORS.values()])

RESET = "\033[0m"
RESET_RE = "\033\\[0m"


def colored(text, color=None, on_color=None, attrs=None):
    if os.getenv("ANSI_COLORS_DISABLED") is None:
        fmt_str = "\033[%dm%s"
        if color is not None:
            text = re.sub(COLORS_RE + "(.*?)" + RESET_RE, r"\1", text)
            text = fmt_str % (COLORS[color], text)
        if on_color is not None:
            text = re.sub(HIGHLIGHTS_RE + "(.*?)" + RESET_RE, r"\1", text)
            text = fmt_str % (HIGHLIGHTS[on_color], text)
        if attrs is not None:
            text = re.sub(ATTRIBUTES_RE + "(.*?)" + RESET_RE, r"\1", text)
            for attr in attrs:
                text = fmt_str % (ATTRIBUTES[attr], text)
        return text + RESET
    else:
        return text
