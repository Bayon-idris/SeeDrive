import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, RED, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)
LINE_COLORS = [(255, 140, 0), (0, 215, 255), (255, 0, 255), (0, 140, 255), (255, 255, 0), (80, 80, 255)]
FLASH_FRAMES = 6            # a line stays white this many frames after a vehicle crossed it


def draw_label(image, text, x, y, scale=0.5):
    """White text with a black outline: readable on any background."""
    cv2.putText(image, text, (x, y), FONT, scale, BLACK, 3)
    cv2.putText(image, text, (x, y), FONT, scale, WHITE, 1)


def draw_tag(image, text, x, y, color, text_color=BLACK, scale=0.45):
    """Text on a filled rectangle, bottom-left corner at (x, y)."""
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 1)
    y = max(y, h + 4)                                   # stay inside the image
    cv2.rectangle(image, (x, y - h - 4), (x + w + 4, y + 2), color, -1)
    cv2.putText(image, text, (x + 2, y - 2), FONT, scale, text_color, 1, cv2.LINE_AA)


def draw_lines(image, lines, last_crossed, frame):
    """Route lines L1, L2, ... each with its color; flashes white when a vehicle crosses it."""
    for i, line in enumerate(lines):
        a, b = [tuple(map(int, p)) for p in line["points"]]
        color = LINE_COLORS[i % len(LINE_COLORS)]
        flash = frame - last_crossed.get(line["name"], -100) <= FLASH_FRAMES
        cv2.line(image, a, b, WHITE if flash else color, 4 if flash else 2, cv2.LINE_AA)
        for p in (a, b):
            cv2.circle(image, p, 4, color, -1)
        draw_tag(image, line["name"], a[0] + 6, a[1] - 6, color, scale=0.55)


def draw_vehicle(image, v, box):
    """Green box, 'ID:12 car 0.87 | 43 km/h' tag, red dot at the center."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
    text = f"ID:{v.id} {v.label}"
    if v.conf is not None:
        text += f" {v.conf:.2f}"
    if v.speed is not None:
        text += f" | {v.speed:.0f} km/h"
    draw_tag(image, text, x1, y1 - 2, GREEN)
    cv2.circle(image, tuple(map(int, v.center)), 4, RED, -1)


def draw_panel(image, rows, x=10, y=10, alpha=0.55, scale=0.5):
    """Semi-transparent dark table. rows = list of lists of strings, first row = header."""
    col_w = [max(cv2.getTextSize(str(r[c]), FONT, scale, 1)[0][0] for r in rows) + 18
             for c in range(len(rows[0]))]
    row_h = 24
    w, h = sum(col_w) + 12, row_h * len(rows) + 10

    roi = image[y:y + h, x:x + w]
    roi[:] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)   # darken behind the table
    for r, row in enumerate(rows):
        cx = x + 8
        for c, cell in enumerate(row):
            color = (0, 215, 255) if r == 0 else WHITE                       # header in yellow
            cv2.putText(image, str(cell), (cx, y + 22 + r * row_h), FONT, scale, color, 1, cv2.LINE_AA)
            cx += col_w[c]
    return image


def route_rows(counts, class_names):
    """Route counts -> table rows: header + one row per route (origin -> destination)."""
    if len(class_names) <= 1:                       # one-class detector: only the total
        rows = [["Route", "Vehicles"]]
        rows += [[f"{o} -> {d}", sum(c.values())] for (o, d), c in counts.items()]
    else:
        rows = [["Route", *class_names, "Total"]]
        rows += [[f"{o} -> {d}", *[c.get(n, 0) for n in class_names], sum(c.values())]
                 for (o, d), c in counts.items()]
    return rows


if __name__ == "__main__":
    from collections import Counter
    from types import SimpleNamespace

    from configs.config import IMAGES_DIR

    image = cv2.imread(str(IMAGES_DIR / "MVI_40892" / "img00001.jpg"))
    lines = [{"name": "L1", "points": [[70, 345], [70, 110]]},
             {"name": "L2", "points": [[110, 120], [620, 175]]}]
    draw_lines(image, lines, {"L2": 0}, frame=0)            # L2 was just crossed -> white

    fake = SimpleNamespace(id=12, label="car", conf=0.87, speed=None, center=(300, 250))
    draw_vehicle(image, fake, (260, 220, 340, 280))

    counts = {("L1", "L2"): Counter(car=5, bus=1), ("L2", "L1"): Counter(car=3)}
    draw_panel(image, route_rows(counts, ["car", "bus", "van", "others"]))

    cv2.imshow("Overlay test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()