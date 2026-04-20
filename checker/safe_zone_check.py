import io
import base64
from PIL import Image, ImageDraw, ImageFilter

SAFE_ZONES = {
    # 1:1 omitted — full image is visible in feed, no safe zone needed
    "9:16": {
        "top": 0.14, "bottom": 0.22, "left": 0.05, "right": 0.05,
        "top_label":    "COVERED BY HEADER (14%)",
        "bottom_label": "COVERED BY UI BAR (22%)",
    },
    "1.91:1": {
        "top": 0.10, "bottom": 0.10, "left": 0.05, "right": 0.05,
        "top_label":    "EDGE CLIPPING (10%)",
        "bottom_label": "EDGE CLIPPING (10%)",
    },
}


def _detect_text_blocks(img, x1, y1, x2, y2, block_size=20):
    """Return list of (bx1,by1,bx2,by2) blocks that contain text-like content."""
    gray = img.convert("L").crop((x1, y1, x2, y2))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    px = list(edges.getdata())
    w = x2 - x1
    h = y2 - y1
    if not px or w <= 0 or h <= 0:
        return []
    avg = sum(px) / len(px)
    threshold = avg * 1.6
    hot_blocks = []
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block = [
                px[cy * w + cx]
                for cy in range(by, min(by + block_size, h))
                for cx in range(bx, min(bx + block_size, w))
                if cy * w + cx < len(px)
            ]
            if block and sum(block) / len(block) > threshold:
                hot_blocks.append((x1 + bx, y1 + by,
                                   x1 + min(bx + block_size, w),
                                   y1 + min(by + block_size, h)))
    return hot_blocks


def _draw_label(draw, text, x, y, bg, fg=(255, 255, 255, 255), font_size=14):
    char_w = int(font_size * 0.65)
    pad = 6
    tw = len(text) * char_w
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + font_size + pad], fill=bg)
    draw.text((x, y), text, fill=fg)


def check_safe_zone(image_path, ratio_name):
    zone = SAFE_ZONES.get(ratio_name)
    if zone is None:
        return None

    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        w, h = img.size

    sx1 = int(w * zone["left"])
    sy1 = int(h * zone["top"])
    sx2 = int(w * (1 - zone["right"]))
    sy2 = int(h * (1 - zone["bottom"]))

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # ── Shade danger zones ──────────────────────────────────────────────────
    danger_bg  = (220, 38, 38, 115)
    draw.rectangle([0,   0,   w,   sy1], fill=danger_bg)   # top
    draw.rectangle([0,   sy2, w,   h  ], fill=danger_bg)   # bottom
    draw.rectangle([0,   sy1, sx1, sy2], fill=danger_bg)   # left
    draw.rectangle([sx2, sy1, w,   sy2], fill=danger_bg)   # right

    # ── Highlight text-like content found inside danger zones ───────────────
    text_in_danger = False
    danger_regions = [
        (0, 0, w, sy1), (0, sy2, w, h),
        (0, sy1, sx1, sy2), (sx2, sy1, w, sy2),
    ]
    for rx1, ry1, rx2, ry2 in danger_regions:
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        blocks = _detect_text_blocks(img, rx1, ry1, rx2, ry2)
        if blocks:
            text_in_danger = True
        for bx1, by1, bx2, by2 in blocks:
            draw.rectangle([bx1, by1, bx2, by2], fill=(255, 165, 0, 160),
                           outline=(255, 120, 0, 255), width=2)

    # ── Safe zone green border ──────────────────────────────────────────────
    draw.rectangle([sx1, sy1, sx2, sy2], outline=(34, 197, 94, 255), width=4)

    # ── Safe zone fill hint (very subtle) ──────────────────────────────────
    safe_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    safe_draw = ImageDraw.Draw(safe_overlay)
    safe_draw.rectangle([sx1, sy1, sx2, sy2], fill=(34, 197, 94, 18))
    overlay = Image.alpha_composite(overlay, safe_overlay)
    draw = ImageDraw.Draw(overlay, "RGBA")

    # ── Labels ─────────────────────────────────────────────────────────────
    # Top danger
    top_text = zone.get("top_label", "DANGER ZONE")
    _draw_label(draw, top_text, 10, max(sy1 - 28, 4),
                bg=(220, 38, 38, 210))

    # Bottom danger
    bot_text = zone.get("bottom_label", "DANGER ZONE")
    _draw_label(draw, bot_text, 10, min(sy2 + 6, h - 26),
                bg=(220, 38, 38, 210))

    # Safe zone label
    _draw_label(draw, "✓  SAFE ZONE — keep all text & logos here",
                sx1 + 8, sy1 + 8, bg=(34, 197, 94, 220))

    # Warning label if text found in danger zones
    if text_in_danger:
        warn = "⚠  Text/content detected here — will be hidden!"
        _draw_label(draw, warn, sx1 + 8, sy1 + 36,
                    bg=(255, 140, 0, 220), fg=(255, 255, 255, 255))

    # ── Encode ─────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    overlay.convert("RGB").save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "safe_zone": {
            "x1": sx1, "y1": sy1, "x2": sx2, "y2": sy2,
            "width_px": sx2 - sx1, "height_px": sy2 - sy1,
        },
        "danger_zone_margins": {
            "top":    f"{zone['top']*100:.0f}%",
            "bottom": f"{zone['bottom']*100:.0f}%",
            "left":   f"{zone['left']*100:.0f}%",
            "right":  f"{zone['right']*100:.0f}%",
        },
        "text_in_danger": text_in_danger,
        "warnings": [],
        "overlay_image": encoded,
    }
