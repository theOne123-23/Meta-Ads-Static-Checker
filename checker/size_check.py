from PIL import Image

# Accepted Meta ad sizes per aspect ratio
ACCEPTED_SIZES = {
    "1:1": {
        "ratio": (1, 1),
        "recommended": (1080, 1080),
        "minimum": (500, 500),
        "maximum": (30720, 30720),
        "placements": ["Feed (Facebook & Instagram)", "Marketplace", "Right Column"],
    },
    "9:16": {
        "ratio": (9, 16),
        "recommended": (1080, 1920),
        "minimum": (500, 889),
        "maximum": (30720, 30720),
        "placements": ["Stories", "Reels", "Facebook Stories", "Instagram Stories"],
    },
    "1.91:1": {
        "ratio": (1.91, 1),
        "recommended": (1200, 628),
        "minimum": (600, 314),
        "maximum": (30720, 30720),
        "placements": ["Facebook Feed (landscape)", "Right Column", "Marketplace"],
    },
}

TOLERANCE = 0.03  # 3% tolerance for ratio matching


def get_ratio(width, height):
    return width / height


def detect_ratio(width, height):
    img_ratio = get_ratio(width, height)
    for name, data in ACCEPTED_SIZES.items():
        target_w, target_h = data["ratio"]
        target_ratio = target_w / target_h
        if abs(img_ratio - target_ratio) <= TOLERANCE:
            return name, data
    return None, None


def check_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        format_ = img.format
        mode = img.mode

    ratio_name, ratio_data = detect_ratio(width, height)

    result = {
        "width": width,
        "height": height,
        "format": format_,
        "mode": mode,
        "detected_ratio": ratio_name,
        "passed": False,
        "issues": [],
        "info": [],
    }

    if ratio_name is None:
        result["issues"].append(
            f"Aspect ratio {width}x{height} does not match any accepted Meta ratio (1:1, 9:16, 1.91:1)."
        )
        return result

    min_w, min_h = ratio_data["minimum"]
    rec_w, rec_h = ratio_data["recommended"]
    placements = ratio_data["placements"]

    result["info"].append(f"Ratio matched: {ratio_name}")
    result["info"].append(f"Suitable for: {', '.join(placements)}")

    if width < min_w or height < min_h:
        result["issues"].append(
            f"Image too small. Minimum for {ratio_name} is {min_w}x{min_h}px. Yours is {width}x{height}px."
        )
    else:
        result["info"].append(f"Size OK — minimum is {min_w}x{min_h}px.")

    if width < rec_w or height < rec_h:
        result["info"].append(
            f"Tip: Recommended size for {ratio_name} is {rec_w}x{rec_h}px for best quality."
        )

    if not result["issues"]:
        result["passed"] = True

    return result
