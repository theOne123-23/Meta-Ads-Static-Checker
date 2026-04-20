import os
import io
import json
import base64
import google.generativeai as genai
from PIL import Image, ImageFilter, ImageStat

SAFE_ZONE_MARGINS = {
    "9:16":   {"top": 0.14, "bottom": 0.22, "left": 0.05, "right": 0.05},
    "1.91:1": {"top": 0.10, "bottom": 0.10, "left": 0.05, "right": 0.05},
}


# ── Image metric helpers ──────────────────────────────────────────────────────

def _sharpness(img_gray):
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    mean = sum(pixels) / len(pixels)
    return sum((p - mean) ** 2 for p in pixels) / len(pixels)


def _contrast(img_gray):
    stat = ImageStat.Stat(img_gray)
    return stat.stddev[0]


def _brightness(img_gray):
    stat = ImageStat.Stat(img_gray)
    return stat.mean[0]


def _saturation(img_rgb):
    r, g, b = img_rgb.split()
    r_data = list(r.getdata())
    g_data = list(g.getdata())
    b_data = list(b.getdata())
    sat_vals = []
    for rv, gv, bv in zip(r_data, g_data, b_data):
        mx = max(rv, gv, bv)
        mn = min(rv, gv, bv)
        sat_vals.append(mx - mn)
    return sum(sat_vals) / len(sat_vals)


def _text_area_pct(img):
    gray = img.convert("L").resize((200, 200))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    avg = sum(pixels) / len(pixels)
    w, h = 200, 200
    block = 10
    text_blocks = total = 0
    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = [pixels[cy * w + cx]
                   for cy in range(y, min(y + block, h))
                   for cx in range(x, min(x + block, w))]
            if blk:
                if sum(blk) / len(blk) > avg * 1.8:
                    text_blocks += 1
                total += 1
    return (text_blocks / total * 100) if total else 0


def _region_activity(img_gray, x1, y1, x2, y2):
    """Fraction of edge-active pixels in a region — high = likely has text/logo."""
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = img_gray.crop((x1, y1, x2, y2)).filter(ImageFilter.FIND_EDGES)
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    threshold = 30
    return sum(1 for p in pixels if p > threshold) / len(pixels)


# ── Safe zone check ───────────────────────────────────────────────────────────

def _safe_zone_check(img_rgb, ratio_name):
    if ratio_name == "1:1" or ratio_name not in SAFE_ZONE_MARGINS:
        return {
            "verdict": "PASS",
            "elements_in_danger_zone": [],
            "explanation": "1:1 ratio — entire image is fully visible in feed. No safe zone restrictions.",
        }

    m = SAFE_ZONE_MARGINS[ratio_name]
    w, h = img_rgb.size
    gray = img_rgb.convert("L")

    sy1 = int(h * m["top"])
    sy2 = int(h * (1 - m["bottom"]))
    sx1 = int(w * m["left"])
    sx2 = int(w * (1 - m["right"]))

    CONTENT_THRESHOLD = 0.06  # >6% active pixels = likely real content, not bg

    zones = {
        "top":    _region_activity(gray, 0,   0,   w,   sy1),
        "bottom": _region_activity(gray, 0,   sy2, w,   h),
        "left":   _region_activity(gray, 0,   sy1, sx1, sy2),
        "right":  _region_activity(gray, sx2, sy1, w,   sy2),
    }

    flagged = [zone for zone, act in zones.items() if act > CONTENT_THRESHOLD]

    zone_desc = {
        "top":    f"Top {int(m['top']*100)}% (platform header / profile area)",
        "bottom": f"Bottom {int(m['bottom']*100)}% (CTA button / engagement bar)",
        "left":   f"Left {int(m['left']*100)}% (edge)",
        "right":  f"Right {int(m['right']*100)}% (edge)",
    }

    if not flagged:
        return {
            "verdict": "PASS",
            "elements_in_danger_zone": [],
            "explanation": "All key content appears to be within the safe zone.",
        }

    elements = [zone_desc[z] for z in flagged]
    return {
        "verdict": "FAIL",
        "elements_in_danger_zone": elements,
        "explanation": f"Content detected in danger zone(s): {', '.join(flagged)}. "
                       "Move key text, logos, or CTAs toward the center of the image.",
    }


# ── Score calculation ─────────────────────────────────────────────────────────

def _quality_score(sharpness, contrast, brightness, saturation, text_pct):
    score = 0
    # Sharpness (25 pts)
    if sharpness >= 150:   score += 25
    elif sharpness >= 80:  score += 18
    elif sharpness >= 40:  score += 10
    else:                  score += 3
    # Contrast (25 pts)
    if contrast >= 65:     score += 25
    elif contrast >= 45:   score += 18
    elif contrast >= 30:   score += 10
    else:                  score += 3
    # Brightness (15 pts) — 90–180 is ideal range
    if 90 <= brightness <= 180:  score += 15
    elif 70 <= brightness <= 200: score += 9
    else:                         score += 3
    # Saturation (15 pts)
    if saturation >= 60:   score += 15
    elif saturation >= 35: score += 9
    else:                  score += 3
    # Text overlay (20 pts)
    if text_pct <= 15:     score += 20
    elif text_pct <= 20:   score += 14
    elif text_pct <= 30:   score += 7
    else:                  score += 0
    return score


def _performance_score(quality, safe_zone_pass):
    score = quality * 0.75
    if safe_zone_pass:
        score += 25
    else:
        score -= 10
    return max(0, min(100, round(score)))


def _winning_verdict(perf):
    if perf >= 75: return "Winning Ad"
    if perf >= 55: return "Promising"
    if perf >= 35: return "Needs Work"
    return "Not Ready"


# ── Feedback generation ───────────────────────────────────────────────────────

def _generate_feedback(metrics, safe_zone, ratio_name):
    sharpness   = metrics["sharpness"]
    contrast    = metrics["contrast"]
    brightness  = metrics["brightness"]
    saturation  = metrics["saturation"]
    text_pct    = metrics["text_pct"]
    sz_pass     = safe_zone["verdict"] == "PASS"
    perf        = metrics["performance_score"]

    strengths, issues, improvements = [], [], []

    # Sharpness
    if sharpness >= 120:
        strengths.append("Image is sharp and high quality — passes Meta's Andromeda creative quality filter.")
    elif sharpness < 60:
        issues.append(f"Image appears soft or blurry (sharpness score: {sharpness:.0f}). Meta penalises low-quality creatives in auction ranking.")
        improvements.append("Reshoot or apply sharpening in Photoshop/Canva before uploading. Aim for crisp, clean edges on text and product.")

    # Contrast
    if contrast >= 55:
        strengths.append("Strong contrast between subject and background — great for thumb-stopping on a scrolling feed.")
    elif contrast < 35:
        issues.append(f"Low contrast (score: {contrast:.0f}). The subject blends into the background, reducing thumb-stop power.")
        improvements.append("Increase contrast: use a darker background behind light subjects (or vice versa). High contrast is one of Meta's top engagement signals.")

    # Brightness
    if 90 <= brightness <= 175:
        strengths.append("Good exposure — image is neither too dark nor washed out.")
    elif brightness < 70:
        issues.append("Image is too dark. Dark ads underperform on mobile screens in bright environments.")
        improvements.append("Brighten the image by +20–30 exposure in your editing tool. Make sure the subject is well-lit.")
    elif brightness > 200:
        issues.append("Image appears overexposed/washed out. Very bright images lose detail and look low-quality.")
        improvements.append("Reduce exposure slightly and add contrast to recover detail in highlights.")

    # Saturation
    if saturation >= 50:
        strengths.append("Vibrant, saturated colors — bold colors are proven to stop the scroll on Meta feeds.")
    elif saturation < 30:
        issues.append("Colors appear dull or muted. Low-saturation ads have lower engagement velocity in Meta's system.")
        improvements.append("Boost color saturation by 15–25% in your design tool. Consider using your brand's boldest colors as the dominant background.")

    # Text overlay
    if text_pct <= 15:
        strengths.append(f"Text overlay is minimal (~{text_pct:.0f}%) — well within Meta's 20% guideline, no delivery penalties.")
    elif text_pct > 20:
        issues.append(f"Text overlay is ~{text_pct:.0f}% of the image — above Meta's 20% limit. This reduces ad delivery reach.")
        improvements.append(f"Remove or reduce on-image text. Move secondary information to the ad headline/primary text fields instead of the image.")

    # Safe zone
    if sz_pass:
        if ratio_name != "1:1":
            strengths.append("All key content is within the safe zone — nothing will be hidden by platform UI.")
    else:
        flagged = safe_zone.get("elements_in_danger_zone", [])
        issues.append(f"Content detected in danger zone(s): {', '.join(flagged)}. This will be covered by Facebook/Instagram UI.")
        improvements.append(f"Shift your text, logo, or CTA button inward — away from the {' and '.join([z.split('(')[0].strip().lower() for z in flagged])} edges — to ensure full visibility on all placements.")

    # Generate analysis paragraph
    analysis = _build_analysis(perf, sharpness, contrast, saturation, text_pct, sz_pass, len(issues), ratio_name)

    return strengths, issues, improvements, analysis


def _build_analysis(perf, sharpness, contrast, saturation, text_pct, sz_pass, issue_count, ratio):
    if perf >= 75:
        line1 = "This ad has strong creative fundamentals — sharp imagery, good contrast, and vibrant colors that are proven to stop the scroll on Meta feeds."
    elif perf >= 55:
        line1 = "This ad has solid potential with some areas that need attention before it can compete at the top of the auction."
    elif perf >= 35:
        line1 = "This ad has meaningful weaknesses that will limit its delivery and performance in Meta's Andromeda system."
    else:
        line1 = "This ad needs significant improvements before it is ready to run — it is likely to score poorly in Meta's creative quality filter."

    if issue_count == 0:
        line2 = "No major policy or quality issues were detected. The creative appears ready for upload."
    elif not sz_pass:
        line2 = "The biggest risk is content in the safe zone danger areas — platform UI will cover important elements, breaking the viewer's experience."
    elif sharpness < 60:
        line2 = "The main weakness is image sharpness — blurry creatives are filtered out early by Meta's Andromeda quality scoring pipeline."
    elif contrast < 35:
        line2 = "The primary issue is low contrast, which reduces thumb-stop power on a crowded mobile feed."
    else:
        line2 = f"There {'is' if issue_count == 1 else 'are'} {issue_count} issue{'s' if issue_count != 1 else ''} flagged — address these before publishing to maximise delivery."

    if perf >= 75:
        line3 = "Consider uploading 3–5 variants of this ad with different angles (lifestyle, social proof, offer-forward) to feed Meta's Andromeda diversification engine and lower your CPMs."
    elif not sz_pass:
        line3 = "Repositioning the key content toward the center of the image is the single highest-impact fix for this creative."
    elif sharpness < 80:
        line3 = "Prioritise image quality first — a sharp, high-resolution version of this same concept will score significantly higher."
    else:
        line3 = "Fixing the flagged issues and re-testing this creative against a high-contrast variant will help identify the best performer for your ad set."

    return f"{line1} {line2} {line3}"


# ── Gemini AI analysis ────────────────────────────────────────────────────────

def _gemini_analyze(image_path: str, ratio_name: str | None) -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        margins = SAFE_ZONE_MARGINS.get(ratio_name or "", {})
        if ratio_name == "1:1" or not margins:
            sz_desc = "This is a 1:1 ratio ad — the ENTIRE image is fully visible. Safe zone = PASS automatically."
        else:
            sz_desc = (
                f"Danger zones for {ratio_name}: top {int(margins['top']*100)}%, "
                f"bottom {int(margins['bottom']*100)}%, left/right {int(margins['left']*100)}%. "
                "Only flag text, logos, CTAs clearly inside danger zones. Backgrounds extending to edges = IGNORE."
            )

        prompt = f"""You are an expert Meta (Facebook/Instagram) ad creative analyst.
Aspect ratio: {ratio_name or 'unknown'}
{sz_desc}

Analyze this static ad image. Return ONLY valid JSON, no other text:
{{
  "safe_zone": {{
    "verdict": "PASS or FAIL",
    "elements_in_danger_zone": ["list elements clearly in danger zones, empty if none"],
    "explanation": "one sentence"
  }},
  "quality": {{
    "score": 0-100,
    "thumb_stop_power": "Strong/Moderate/Weak",
    "text_clarity": "Clear/Borderline/Poor",
    "value_prop_visible": true or false,
    "contrast": "High/Medium/Low",
    "clutter_level": "Clean/Moderate/Cluttered"
  }},
  "performance_score": 0-100,
  "winning_verdict": "Winning Ad/Promising/Needs Work/Not Ready",
  "is_winning_ad": true or false,
  "strengths": ["specific strength 1", "specific strength 2"],
  "issues": ["specific issue 1 (if any)"],
  "improvements": ["specific tip 1", "specific tip 2", "specific tip 3"],
  "analysis": "3 sentences: what works, biggest risk, top recommendation — be specific about THIS ad"
}}"""

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()},
            prompt,
        ])

        raw = response.text.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(raw)

    except Exception:
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_ad(image_path: str, ratio_name: str | None) -> dict:
    # Try Gemini first — falls back to rule-based if key not set or call fails
    gemini = _gemini_analyze(image_path, ratio_name)
    if gemini:
        return gemini

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_gray = img.convert("L")

        sharpness  = round(_sharpness(img_gray), 1)
        contrast   = round(_contrast(img_gray), 1)
        brightness = round(_brightness(img_gray), 1)
        saturation = round(_saturation(img), 1)
        text_pct   = round(_text_area_pct(img), 1)

        safe_zone  = _safe_zone_check(img, ratio_name)

    sz_pass    = safe_zone["verdict"] == "PASS"
    q_score    = _quality_score(sharpness, contrast, brightness, saturation, text_pct)
    perf_score = _performance_score(q_score, sz_pass)
    verdict    = _winning_verdict(perf_score)

    metrics = {
        "sharpness": sharpness, "contrast": contrast,
        "brightness": brightness, "saturation": saturation,
        "text_pct": text_pct, "performance_score": perf_score,
    }

    strengths, issues, improvements, analysis = _generate_feedback(metrics, safe_zone, ratio_name)

    thumb_stop  = "Strong" if saturation >= 50 and contrast >= 55 else "Moderate" if saturation >= 30 or contrast >= 40 else "Weak"
    text_clear  = "Clear" if sharpness >= 80 else "Borderline" if sharpness >= 40 else "Poor"
    contrast_lv = "High" if contrast >= 65 else "Medium" if contrast >= 35 else "Low"
    clutter     = "Clean" if text_pct <= 15 else "Moderate" if text_pct <= 25 else "Cluttered"

    return {
        "safe_zone":        safe_zone,
        "quality": {
            "score":              q_score,
            "thumb_stop_power":   thumb_stop,
            "text_clarity":       text_clear,
            "value_prop_visible": text_pct >= 5,
            "contrast":           contrast_lv,
            "clutter_level":      clutter,
        },
        "performance_score": perf_score,
        "winning_verdict":   verdict,
        "is_winning_ad":     perf_score >= 75,
        "strengths":         strengths,
        "issues":            issues,
        "improvements":      improvements,
        "analysis":          analysis,
    }
