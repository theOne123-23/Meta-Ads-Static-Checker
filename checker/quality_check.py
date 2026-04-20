from PIL import Image, ImageFilter
import statistics

SHARPNESS_THRESHOLD = 80    # Laplacian variance below this = blurry
CONTRAST_THRESHOLD  = 45    # Std deviation of gray below this = low contrast
TEXT_AREA_MAX       = 0.20  # Max 20% of image area should be text-heavy


def _laplacian_variance(gray_img):
    """Higher = sharper image."""
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    if not pixels:
        return 0
    mean = sum(pixels) / len(pixels)
    return sum((p - mean) ** 2 for p in pixels) / len(pixels)


def _contrast_score(gray_img):
    """Standard deviation of pixel brightness. Higher = more contrast."""
    pixels = list(gray_img.getdata())
    if not pixels:
        return 0
    mean = sum(pixels) / len(pixels)
    variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
    return variance ** 0.5


def _estimate_text_area(img):
    """
    Rough estimate of how much of the image is text/graphics overlay.
    Uses edge density in small blocks — high edge density = likely text region.
    Returns fraction of total image area estimated as text.
    """
    gray = img.convert("L").resize((200, 200))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    avg = sum(pixels) / len(pixels)

    block_size = 10
    width, height = 200, 200
    text_blocks = 0
    total_blocks = 0

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = [
                pixels[cy * width + cx]
                for cy in range(y, min(y + block_size, height))
                for cx in range(x, min(x + block_size, width))
            ]
            if not block:
                continue
            block_mean = sum(block) / len(block)
            if block_mean > avg * 1.8:
                text_blocks += 1
            total_blocks += 1

    return text_blocks / total_blocks if total_blocks > 0 else 0


def check_quality(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        gray = img.convert("L")

        sharpness  = _laplacian_variance(gray)
        contrast   = _contrast_score(gray)
        text_ratio = _estimate_text_area(img)

    issues = []
    info   = []
    passed = True

    # Sharpness
    if sharpness < SHARPNESS_THRESHOLD:
        issues.append(
            f"Image appears blurry or soft (sharpness score: {sharpness:.1f}). "
            "Meta's Andromeda quality filter penalises low-resolution/blurry creatives."
        )
        passed = False
    else:
        info.append(f"Image sharpness OK (score: {sharpness:.1f}).")

    # Contrast
    if contrast < CONTRAST_THRESHOLD:
        issues.append(
            f"Image has low contrast (score: {contrast:.1f}). "
            "High contrast between subject and background improves thumb-stop rate."
        )
        passed = False
    else:
        info.append(f"Image contrast OK (score: {contrast:.1f}).")

    # Text overlay
    text_pct = text_ratio * 100
    if text_ratio > TEXT_AREA_MAX:
        issues.append(
            f"Estimated text/graphics area is ~{text_pct:.1f}% of the image — "
            "Meta policy requires text overlay under 20%. Heavy text overlays also "
            "reduce Andromeda Ad Quality Score."
        )
        passed = False
    else:
        info.append(
            f"Text overlay estimate: ~{text_pct:.1f}% (under the 20% limit)."
        )

    return {
        "passed": passed,
        "sharpness_score": round(sharpness, 1),
        "contrast_score":  round(contrast, 1),
        "text_area_pct":   round(text_pct, 1),
        "issues": issues,
        "info":   info,
    }
