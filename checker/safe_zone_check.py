import io
import base64
from PIL import Image, ImageDraw, ImageFont

# Safe zone definitions as percentages of image dimensions.
# Values = how much to EXCLUDE (danger zone) from each edge.
# Everything INSIDE the remaining area is the "safe zone".
SAFE_ZONES = {
    "1:1": {
        "top":    0.05,
        "bottom": 0.05,
        "left":   0.05,
        "right":  0.05,
        "notes": [
            "Top/bottom 5%: may be clipped by feed UI borders.",
            "Keep logos, faces, and key text inside the safe zone.",
        ],
    },
    "9:16": {
        "top":    0.14,
        "bottom": 0.22,
        "left":   0.05,
        "right":  0.05,
        "notes": [
            "Top 14%: covered by Reels/Stories header and page name.",
            "Bottom 22%: covered by profile, CTA button, engagement icons.",
            "Left/right 5%: edge clipping risk.",
        ],
    },
    "1.91:1": {
        "top":    0.10,
        "bottom": 0.10,
        "left":   0.05,
        "right":  0.05,
        "notes": [
            "All edges have a ~10% margin for landscape placements.",
            "Keep key content centered in the safe zone.",
        ],
    },
}

# How 'active' a pixel must be to count as foreground content (0-255 range)
ACTIVITY_THRESHOLD = 20
# What fraction of danger-zone pixels being 'active' triggers a warning
CONTENT_WARNING_RATIO = 0.04


def _pixel_activity(img_gray, x1, y1, x2, y2):
    """Return fraction of pixels in a region that differ from their surroundings."""
    region = img_gray.crop((x1, y1, x2, y2))
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    avg = sum(pixels) / len(pixels)
    active = sum(1 for p in pixels if abs(p - avg) > ACTIVITY_THRESHOLD)
    return active / len(pixels)


def check_safe_zone(image_path, ratio_name):
    zone = SAFE_ZONES.get(ratio_name)
    if zone is None:
        return None

    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        width, height = img.size

    # Compute safe zone pixel bounds
    safe_x1 = int(width  * zone["left"])
    safe_y1 = int(height * zone["top"])
    safe_x2 = int(width  * (1 - zone["right"]))
    safe_y2 = int(height * (1 - zone["bottom"]))

    # Analyse danger zones for content presence
    gray = img.convert("L")
    danger_regions = {
        "top":    (0,       0,       width,   safe_y1),
        "bottom": (0,       safe_y2, width,   height),
        "left":   (0,       safe_y1, safe_x1, safe_y2),
        "right":  (safe_x2, safe_y1, width,   safe_y2),
    }

    warnings = []
    for region_name, (rx1, ry1, rx2, ry2) in danger_regions.items():
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        activity = _pixel_activity(gray, rx1, ry1, rx2, ry2)
        if activity > CONTENT_WARNING_RATIO:
            warnings.append(
                f"Content detected in the {region_name} danger zone "
                f"({activity*100:.1f}% active pixels). "
                "Important elements here may be hidden by platform UI."
            )

    # Build annotated overlay image
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Shade danger zones with semi-transparent red
    danger_color = (220, 38, 38, 100)
    draw.rectangle([0, 0, width, safe_y1],          fill=danger_color)   # top
    draw.rectangle([0, safe_y2, width, height],      fill=danger_color)   # bottom
    draw.rectangle([0, safe_y1, safe_x1, safe_y2],   fill=danger_color)   # left
    draw.rectangle([safe_x2, safe_y1, width, safe_y2], fill=danger_color) # right

    # Draw safe zone border in green
    border = 4
    draw.rectangle(
        [safe_x1, safe_y1, safe_x2, safe_y2],
        outline=(34, 197, 94, 255),
        width=border,
    )

    # Label safe zone
    label = "SAFE ZONE"
    label_x = safe_x1 + 10
    label_y = safe_y1 + 10
    draw.rectangle(
        [label_x - 4, label_y - 4, label_x + len(label) * 9 + 4, label_y + 22],
        fill=(34, 197, 94, 200),
    )
    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255))

    # Encode overlay as base64 PNG for frontend display
    buf = io.BytesIO()
    overlay.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "safe_zone": {
            "x1": safe_x1, "y1": safe_y1,
            "x2": safe_x2, "y2": safe_y2,
            "width_px":  safe_x2 - safe_x1,
            "height_px": safe_y2 - safe_y1,
        },
        "danger_zone_margins": {
            "top":    f"{zone['top']*100:.0f}%",
            "bottom": f"{zone['bottom']*100:.0f}%",
            "left":   f"{zone['left']*100:.0f}%",
            "right":  f"{zone['right']*100:.0f}%",
        },
        "warnings": warnings,
        "notes": zone["notes"],
        "overlay_image": encoded,
    }
