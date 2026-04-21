import os
import io
import sys
import json
import base64
import urllib.request
import urllib.error
from PIL import Image, ImageFilter, ImageStat

SAFE_ZONE_MARGINS = {
    "9:16":   {"top": 0.14, "bottom": 0.22, "left": 0.05, "right": 0.05},
    "1.91:1": {"top": 0.10, "bottom": 0.10, "left": 0.05, "right": 0.05},
}

# Prohibited health/medical claim words
PROHIBITED_HEALTH_WORDS = [
    "cure", "cures", "cured", "heal", "heals", "healed",
    "treat", "treats", "treatment", "diagnose", "prevent",
    "guaranteed results", "100% effective", "instant relief",
    "clinically proven", "medically proven", "doctor approved",
    "fix", "fixes", "fixed", "eliminate", "eliminates",
]

# Personal attribute trigger phrases
PERSONAL_ATTRIBUTE_PHRASES = [
    "are you overweight", "struggling with", "do you have",
    "people like you", "if you have", "suffering from",
    "do you suffer", "are you struggling", "your weight",
    "your body", "your skin condition", "your disability",
    "your health condition",
]

# Engagement bait phrases
ENGAGEMENT_BAIT = [
    "like if", "share if", "comment yes", "tag a friend",
    "click like", "type yes", "tag someone",
]


# ── Image metric helpers ──────────────────────────────────────────────────────

def _sharpness(img_gray):
    edges = img_gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    mean = sum(pixels) / len(pixels)
    return sum((p - mean) ** 2 for p in pixels) / len(pixels)


def _contrast(img_gray):
    return ImageStat.Stat(img_gray).stddev[0]


def _brightness(img_gray):
    return ImageStat.Stat(img_gray).mean[0]


def _saturation(img_rgb):
    r, g, b = img_rgb.split()
    sat_vals = [max(rv, gv, bv) - min(rv, gv, bv)
                for rv, gv, bv in zip(r.getdata(), g.getdata(), b.getdata())]
    return sum(sat_vals) / len(sat_vals)


def _text_area_pct(img):
    gray = img.convert("L").resize((200, 200))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    pixels = list(edges.getdata())
    avg = sum(pixels) / len(pixels)
    w, h, block = 200, 200, 10
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
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = img_gray.crop((x1, y1, x2, y2)).filter(ImageFilter.FIND_EDGES)
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    return sum(1 for p in pixels if p > 30) / len(pixels)


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
    sy1, sy2 = int(h * m["top"]), int(h * (1 - m["bottom"]))
    sx1, sx2 = int(w * m["left"]), int(w * (1 - m["right"]))

    zones = {
        "top":    _region_activity(gray, 0,   0,   w,   sy1),
        "bottom": _region_activity(gray, 0,   sy2, w,   h),
        "left":   _region_activity(gray, 0,   sy1, sx1, sy2),
        "right":  _region_activity(gray, sx2, sy1, w,   sy2),
    }
    flagged = [z for z, a in zones.items() if a > 0.06]
    zone_desc = {
        "top":    f"Top {int(m['top']*100)}% (platform header / profile area)",
        "bottom": f"Bottom {int(m['bottom']*100)}% (CTA button / engagement bar)",
        "left":   f"Left {int(m['left']*100)}% (edge)",
        "right":  f"Right {int(m['right']*100)}% (edge)",
    }

    if not flagged:
        return {"verdict": "PASS", "elements_in_danger_zone": [],
                "explanation": "All key content appears to be within the safe zone."}

    return {
        "verdict": "FAIL",
        "elements_in_danger_zone": [zone_desc[z] for z in flagged],
        "explanation": f"Content detected in danger zone(s): {', '.join(flagged)}. "
                       "Move key text, logos, or CTAs toward the center of the image.",
    }


# ── Rule-based text policy check ─────────────────────────────────────────────

_BEFORE_AFTER_PATTERNS = [
    "before and after", "before & after", "before/after",
    "post and pre", "pre and post", "pre treatment", "post treatment",
    "before treatment", "after treatment", "before the treatment", "after the treatment",
    "before photo", "after photo", "before pic", "after pic",
    "transformation photo", "transformation result",
    "results after", "weeks result", "days result",
    "week transformation", "day transformation",
    "see the difference", "see real results", "real results",
    "client results", "patient results", "before vs after", "before vs. after",
]

_PERSONAL_ATTRIBUTE_PATTERNS = [
    "are you overweight", "are you fat", "are you obese",
    "struggling with your weight", "struggling with acne", "struggling with",
    "do you suffer", "suffering from", "if you have",
    "people like you", "your body problem", "your skin condition",
    "your health condition", "do you have diabetes", "are you diabetic",
    "your weight problem",
]

_HEALTH_CLAIM_PATTERNS = [
    "cures ", "cure your", "cured my", "heals ", "heal your",
    "treats ", "treat your", "treats acne", "treats wrinkles",
    "diagnose", "prevents ", "prevent your",
    "guaranteed results", "100% effective", "100% results",
    "instant relief", "instant results",
    "clinically proven", "medically proven", "doctor approved",
    "fda approved", "dermatologist approved", "board certified",
    "eliminates acne", "eliminates wrinkles", "fixes acne", "fixes wrinkles",
    "remove cellulite", "removes cellulite",
]

_ENGAGEMENT_BAIT_PATTERNS = [
    "like if", "share if", "comment yes", "tag a friend",
    "tag someone", "click like", "type yes", "type amen",
    "share this if", "tag your friends", "comment below if",
]

_NEGATIVE_BODY_PATTERNS = [
    "hate your body", "ashamed of your", "embarrassed by your",
    "feel ugly", "look ugly", "body shaming",
]

_AGE_GATE_PATTERNS = [
    "weight loss", "fat burner", "botox", "filler", "lip filler",
    "cosmetic procedure", "cosmetic surgery", "dietary supplement",
    "weight loss pill", "diet pill", "slimming", "anti-aging", "anti-ageing",
]


def _text_policy_check(ad_copy: str) -> dict:
    """Comprehensive rule-based policy check on ad copy text."""
    text = ad_copy.lower()
    violations, warnings = [], []

    # Before/after
    ba_found = next((p for p in _BEFORE_AFTER_PATTERNS if p in text), None)
    before_after_detected = ba_found is not None
    if before_after_detected:
        violations.append(
            f"Before/after language detected: \"{ba_found}\" — prohibited for health, "
            "cosmetic, anti-aging, and weight-loss ads under Meta Advertising Policies."
        )

    # Personal attributes
    pa_found = next((p for p in _PERSONAL_ATTRIBUTE_PATTERNS if p in text), None)
    personal_issue = pa_found is not None
    if personal_issue:
        violations.append(
            f"Personal attribute language: \"{pa_found}\" — ads must not imply knowledge "
            "of personal health, weight, or physical characteristics."
        )

    # Health claims
    hc_found = next((p for p in _HEALTH_CLAIM_PATTERNS if p in text), None)
    health_issue = hc_found is not None
    if health_issue:
        violations.append(
            f"Prohibited health claim: \"{hc_found}\" — replace with "
            "\"may help support\" or \"designed to promote\"."
        )

    # Engagement bait
    bait_found = next((p for p in _ENGAGEMENT_BAIT_PATTERNS if p in text), None)
    bait_issue = bait_found is not None
    if bait_issue:
        violations.append(
            f"Engagement bait: \"{bait_found}\" — Meta prohibits asking users to like, share, or tag."
        )

    # Negative body image
    nbi_found = next((p for p in _NEGATIVE_BODY_PATTERNS if p in text), None)
    nbi_issue = nbi_found is not None
    if nbi_issue:
        warnings.append(
            f"Possible negative body image language: \"{nbi_found}\" — review to ensure "
            "it doesn't exploit insecurities."
        )

    # Age gate
    age_found = next((p for p in _AGE_GATE_PATTERNS if p in text), None)
    age_gate = age_found is not None
    if age_gate:
        warnings.append(
            f"Category \"{age_found}\" typically requires 18+ audience targeting on Meta."
        )

    if violations:
        overall, is_compliant = "VIOLATION", False
    elif warnings:
        overall, is_compliant = "REVIEW REQUIRED", None
    else:
        overall, is_compliant = "COMPLIANT", True
        if ad_copy.strip():
            warnings.append(
                "No policy violations found in ad copy. "
                "Also review image visuals for before/after content not mentioned in text."
            )

    if before_after_detected:
        ba_note = (
            f"Before/after language \"{ba_found}\" detected. "
            "This is PROHIBITED for health, cosmetic, anti-aging, and weight-loss ads on Meta."
        )
    elif ad_copy.strip():
        ba_note = "No before/after language detected in the ad copy you provided."
    else:
        ba_note = "No ad copy entered. Paste your headline/caption above to check automatically."

    return {
        "overall": overall,
        "is_compliant": is_compliant,
        "before_after_detected": before_after_detected,
        "before_after_note": ba_note,
        "personal_attributes_issue": personal_issue,
        "personal_attributes_note": (f'Found: "{pa_found}"' if pa_found else None),
        "health_claim_issue": health_issue,
        "health_claim_note": (f'Found: "{hc_found}"' if hc_found else None),
        "negative_body_image_issue": nbi_issue,
        "negative_body_image_note": (f'Found: "{nbi_found}"' if nbi_found else None),
        "engagement_bait_issue": bait_issue,
        "age_gate_recommended": age_gate,
        "age_gate_reason": (f'Category "{age_found}" requires 18+ targeting' if age_found else None),
        "violations": violations,
        "warnings": warnings,
    }


# ── Score calculation ─────────────────────────────────────────────────────────

def _quality_score(sharpness, contrast, brightness, saturation, text_pct):
    score = 0
    score += 25 if sharpness >= 150 else 18 if sharpness >= 80 else 10 if sharpness >= 40 else 3
    score += 25 if contrast >= 65 else 18 if contrast >= 45 else 10 if contrast >= 30 else 3
    score += 15 if 90 <= brightness <= 180 else 9 if 70 <= brightness <= 200 else 3
    score += 15 if saturation >= 60 else 9 if saturation >= 35 else 3
    score += 20 if text_pct <= 15 else 14 if text_pct <= 20 else 7 if text_pct <= 30 else 0
    return score


def _performance_score(quality, safe_zone_pass):
    score = quality * 0.75 + (25 if safe_zone_pass else -10)
    return max(0, min(100, round(score)))


def _winning_verdict(perf):
    if perf >= 75: return "Winning Ad"
    if perf >= 55: return "Promising"
    if perf >= 35: return "Needs Work"
    return "Not Ready"


# ── Feedback generation ───────────────────────────────────────────────────────

def _generate_feedback(metrics, safe_zone, ratio_name):
    sharpness, contrast = metrics["sharpness"], metrics["contrast"]
    brightness, saturation = metrics["brightness"], metrics["saturation"]
    text_pct, perf = metrics["text_pct"], metrics["performance_score"]
    sz_pass = safe_zone["verdict"] == "PASS"

    strengths, issues, improvements = [], [], []

    if sharpness >= 120:
        strengths.append("Image is sharp and high quality — passes Meta's Andromeda creative quality filter.")
    elif sharpness < 60:
        issues.append(f"Image appears soft or blurry (sharpness: {sharpness:.0f}). Meta penalises low-quality creatives.")
        improvements.append("Reshoot or apply sharpening in Photoshop/Canva. Aim for crisp, clean edges on text and product.")

    if contrast >= 55:
        strengths.append("Strong contrast between subject and background — great for thumb-stopping on a scrolling feed.")
    elif contrast < 35:
        issues.append(f"Low contrast ({contrast:.0f}). Subject blends into background, reducing thumb-stop power.")
        improvements.append("Use a darker background behind light subjects (or vice versa). High contrast is one of Meta's top engagement signals.")

    if 90 <= brightness <= 175:
        strengths.append("Good exposure — image is neither too dark nor washed out.")
    elif brightness < 70:
        issues.append("Image is too dark. Dark ads underperform on mobile screens.")
        improvements.append("Brighten by +20–30 exposure in your editing tool. Ensure the subject is well-lit.")
    elif brightness > 200:
        issues.append("Image is overexposed/washed out. Bright images lose detail and look low-quality.")
        improvements.append("Reduce exposure slightly and add contrast to recover highlight detail.")

    if saturation >= 50:
        strengths.append("Vibrant, saturated colors — bold colors stop the scroll on Meta feeds.")
    elif saturation < 30:
        issues.append("Colors appear dull or muted. Low-saturation ads have lower engagement velocity.")
        improvements.append("Boost color saturation by 15–25%. Use your brand's boldest colors as the dominant background.")

    if text_pct <= 15:
        strengths.append(f"Minimal text overlay (~{text_pct:.0f}%) — within Meta's 20% guideline, no delivery penalty.")
    elif text_pct > 20:
        issues.append(f"Text overlay ~{text_pct:.0f}% — above Meta's 20% limit. This reduces ad delivery reach.")
        improvements.append("Reduce on-image text. Move secondary info to the headline/primary text fields instead.")

    if sz_pass:
        if ratio_name != "1:1":
            strengths.append("All key content is within the safe zone — nothing will be hidden by platform UI.")
    else:
        flagged = safe_zone.get("elements_in_danger_zone", [])
        issues.append(f"Content in danger zone(s): {', '.join(flagged)}. Will be covered by Facebook/Instagram UI.")
        improvements.append(f"Shift text, logo, or CTA inward from the {' and '.join([z.split('(')[0].strip().lower() for z in flagged])} edges.")

    analysis = _build_analysis(perf, sharpness, contrast, saturation, text_pct, sz_pass, len(issues), ratio_name)
    return strengths, issues, improvements, analysis


def _build_analysis(perf, sharpness, contrast, saturation, text_pct, sz_pass, issue_count, ratio):
    if perf >= 75:
        l1 = "This ad has strong creative fundamentals — sharp imagery, good contrast, and vibrant colors that stop the scroll on Meta feeds."
    elif perf >= 55:
        l1 = "This ad has solid potential with some areas that need attention before it can compete at the top of the auction."
    elif perf >= 35:
        l1 = "This ad has meaningful weaknesses that will limit its delivery and performance in Meta's Andromeda system."
    else:
        l1 = "This ad needs significant improvements before it is ready to run — it will likely score poorly in Meta's creative quality filter."

    if issue_count == 0:
        l2 = "No major quality issues were detected. The creative appears ready for upload."
    elif not sz_pass:
        l2 = "The biggest risk is content in the safe zone danger areas — platform UI will cover key elements."
    elif sharpness < 60:
        l2 = "The main weakness is sharpness — blurry creatives are filtered early by Meta's Andromeda quality pipeline."
    elif contrast < 35:
        l2 = "The primary issue is low contrast, reducing thumb-stop power on a crowded mobile feed."
    else:
        l2 = f"There {'is' if issue_count == 1 else 'are'} {issue_count} issue{'s' if issue_count != 1 else ''} flagged — address these before publishing."

    if perf >= 75:
        l3 = "Upload 3–5 variants with different angles (lifestyle, social proof, offer-forward) to feed Meta's Andromeda diversification engine and lower CPMs."
    elif not sz_pass:
        l3 = "Repositioning key content toward the center is the single highest-impact fix for this creative."
    elif sharpness < 80:
        l3 = "Prioritise image quality first — a sharp, high-res version of this concept will score significantly higher."
    else:
        l3 = "Fix the flagged issues and re-test against a high-contrast variant to find the best performer."

    return f"{l1} {l2} {l3}"


# ── Gemini AI analysis ────────────────────────────────────────────────────────

def _gemini_analyze(image_path: str, ratio_name: str | None, ad_copy: str = "") -> dict | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        margins = SAFE_ZONE_MARGINS.get(ratio_name or "", {})
        if ratio_name == "1:1" or not margins:
            sz_desc = "1:1 ratio — ENTIRE image is fully visible. Safe zone = PASS automatically."
        else:
            sz_desc = (
                f"Danger zones for {ratio_name}: top {int(margins['top']*100)}%, "
                f"bottom {int(margins['bottom']*100)}%, left/right {int(margins['left']*100)}%. "
                "Flag only text/logos/CTAs clearly inside danger zones. Backgrounds extending to edges = IGNORE."
            )

        copy_section = f"\nAD COPY / TEXT PROVIDED BY USER:\n\"\"\"\n{ad_copy}\n\"\"\"\nAnalyze this text for policy violations in addition to the image.\n" if ad_copy.strip() else ""

        prompt = f"""You are an expert Meta (Facebook/Instagram) ad compliance analyst and creative strategist.
Aspect ratio: {ratio_name or 'unknown'}
{sz_desc}{copy_section}

META AD POLICY RULES TO ENFORCE:
1. BEFORE/AFTER: Prohibited for weight loss, cosmetic procedures (Botox, fillers, anti-aging, skin treatments, acne). Allowed ONLY for: non-permanent cosmetics (makeup, hair extensions), fitness classes, digital editing apps.
2. PERSONAL ATTRIBUTES: Must not imply knowledge of race, health, weight, age, disability, financial status, sexual orientation. Flag language like "Are you overweight?", "Struggling with acne?", "People like you..."
3. HEALTH CLAIMS: Prohibited words: cure, treat, heal, fix, diagnose, prevent, guaranteed results, 100% effective, instant relief, clinically proven (unless verified). Use "may help support", "designed to promote" instead.
4. NEGATIVE SELF-PERCEPTION: No body-shaming, zoomed-in images of body conditions, content implying a "perfect body", or messaging that exploits insecurities.
5. ENGAGEMENT BAIT: No "Like if...", "Share if...", "Tag a friend", "Comment YES if..."
6. TEXT OVERLAY: Should not exceed ~20% of image area. Should not obstruct the visual.

META BEST PRACTICES:
- High resolution, not overly photoshopped
- Show brand or logo for credibility
- Show people using product/service in realistic settings
- Modern clean font, large enough, high contrast against background
- Tight crop around the important part
- Appealing colors appropriate for content (bright for sales, calming pastels for spa/wellness)
- 18+ age gate required for: weight loss, dietary supplements, cosmetic procedures

Analyze this ad image and return ONLY valid JSON, no markdown, no explanation:
{{
  "policy_compliance": {{
    "overall": "COMPLIANT or VIOLATION or REVIEW REQUIRED",
    "is_compliant": true or false,
    "before_after_detected": true or false,
    "before_after_note": "null or specific explanation of what was detected and why it is/isn't a violation",
    "personal_attributes_issue": true or false,
    "personal_attributes_note": "null or specific issue found",
    "health_claim_issue": true or false,
    "health_claim_note": "null or specific prohibited words/claims found in the image",
    "negative_body_image_issue": true or false,
    "negative_body_image_note": "null or specific issue",
    "engagement_bait_issue": true or false,
    "age_gate_recommended": true or false,
    "age_gate_reason": "null or why 18+ targeting is recommended",
    "violations": ["list each specific policy violation, empty if none"],
    "warnings": ["yellow flags that could trigger review but may be borderline"]
  }},
  "best_practices": {{
    "has_brand_logo": true or false,
    "shows_product_in_use": true or false,
    "text_is_readable": true or false,
    "image_is_high_quality": true or false,
    "shows_real_people": true or false,
    "color_appeal": "Strong/Moderate/Weak",
    "message_clarity": "Clear/Moderate/Unclear",
    "recommendations": ["specific actionable tip 1", "specific actionable tip 2"]
  }},
  "safe_zone": {{
    "verdict": "PASS or FAIL",
    "elements_in_danger_zone": [],
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
  "issues": ["specific issue 1"],
  "improvements": ["specific tip 1", "specific tip 2", "specific tip 3"],
  "analysis": "3 sentences: what works, biggest risk, top recommendation — be specific about THIS ad"
}}"""

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Detect MIME type from file header
        mime = "image/jpeg"
        if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            mime = "image/png"
        elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
            mime = "image/gif"
        elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
            mime = "image/webp"

        payload = json.dumps({
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": mime,
                                     "data": base64.b64encode(img_bytes).decode()}},
                    {"text": prompt},
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
        }).encode("utf-8")

        # Only try models confirmed to exist on v1beta (others all return 404)
        models_to_try = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ]
        last_error = None
        for model in models_to_try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={api_key}")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                print(f"[Gemini] success with model: {model}", file=sys.stderr)
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                return json.loads(raw)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                print(f"[Gemini] {model} HTTP {e.code}: {body}", file=sys.stderr)
                last_error = e
                continue
            except json.JSONDecodeError as e:
                print(f"[Gemini] {model} JSON parse error: {e}", file=sys.stderr)
                return None
            except Exception as e:
                print(f"[Gemini] {model} error: {type(e).__name__}: {e}", file=sys.stderr)
                last_error = e
                continue

        print(f"[Gemini] all models failed, last error: {last_error}", file=sys.stderr)
        return None

    except Exception as e:
        print(f"[Gemini] setup error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ── OpenRouter vision analysis (free tier) ───────────────────────────────────

def _openrouter_free_vision_models(api_key: str) -> list:
    """Query OpenRouter's model list and return currently available free vision models."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            modality = m.get("architecture", {}).get("modality", "")
            is_free = str(pricing.get("prompt", "1")) == "0" and str(pricing.get("completion", "1")) == "0"
            has_vision = "image" in modality.lower()
            if is_free and has_vision:
                models.append(m["id"])
        print(f"[OpenRouter] {len(models)} free vision models found: {models}", file=sys.stderr)
        return models
    except Exception as e:
        print(f"[OpenRouter] model discovery failed: {e}", file=sys.stderr)
        return []

def _openrouter_analyze(image_path: str, ratio_name: str | None) -> dict | None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        margins = SAFE_ZONE_MARGINS.get(ratio_name or "", {})
        if ratio_name == "1:1" or not margins:
            sz_desc = "1:1 ratio — ENTIRE image is fully visible. Safe zone = PASS automatically."
        else:
            sz_desc = (
                f"Danger zones for {ratio_name}: top {int(margins['top']*100)}%, "
                f"bottom {int(margins['bottom']*100)}%, left/right {int(margins['left']*100)}%. "
                "Flag only text/logos/CTAs clearly inside danger zones."
            )

        prompt = f"""You are an expert Meta (Facebook/Instagram) ad compliance analyst and creative strategist.
Aspect ratio: {ratio_name or 'unknown'}
{sz_desc}

META AD POLICY RULES:
1. BEFORE/AFTER: Prohibited for weight loss, cosmetic procedures, anti-aging, skin/acne treatments.
2. PERSONAL ATTRIBUTES: Must not imply knowledge of health, weight, age, disability.
3. HEALTH CLAIMS: Prohibited: cure, treat, heal, guaranteed results, clinically proven (unless verified).
4. NEGATIVE SELF-PERCEPTION: No body-shaming or content exploiting insecurities.
5. ENGAGEMENT BAIT: No "Like if...", "Tag a friend", "Comment YES".
6. TEXT OVERLAY: Should not exceed ~20% of image area.
18+ age gate required for: weight loss, supplements, cosmetic procedures.

Return ONLY valid JSON, no markdown:
{{"policy_compliance":{{"overall":"COMPLIANT or VIOLATION or REVIEW REQUIRED","is_compliant":true or false,"before_after_detected":true or false,"before_after_note":"explanation or null","personal_attributes_issue":true or false,"personal_attributes_note":"issue or null","health_claim_issue":true or false,"health_claim_note":"issue or null","negative_body_image_issue":true or false,"negative_body_image_note":"issue or null","engagement_bait_issue":true or false,"age_gate_recommended":true or false,"age_gate_reason":"reason or null","violations":[],"warnings":[]}},"best_practices":{{"has_brand_logo":true or false,"shows_product_in_use":true or false,"text_is_readable":true or false,"image_is_high_quality":true or false,"shows_real_people":true or false,"color_appeal":"Strong/Moderate/Weak","message_clarity":"Clear/Moderate/Unclear","recommendations":[]}},"safe_zone":{{"verdict":"PASS or FAIL","elements_in_danger_zone":[],"explanation":"one sentence"}},"quality":{{"score":0-100,"thumb_stop_power":"Strong/Moderate/Weak","text_clarity":"Clear/Borderline/Poor","value_prop_visible":true or false,"contrast":"High/Medium/Low","clutter_level":"Clean/Moderate/Cluttered"}},"performance_score":0-100,"winning_verdict":"Winning Ad/Promising/Needs Work/Not Ready","is_winning_ad":true or false,"strengths":[],"issues":[],"improvements":[],"analysis":"3 sentences specific to this ad"}}"""

        with open(image_path, "rb") as f:
            img_bytes = f.read()

        # Resize to max 1024px to keep token usage low
        try:
            from PIL import Image as PilImage
            import io as _io
            pil = PilImage.open(_io.BytesIO(img_bytes)).convert("RGB")
            if max(pil.size) > 1024:
                pil.thumbnail((1024, 1024), PilImage.LANCZOS)
            buf = _io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
        except Exception:
            pass

        b64 = base64.b64encode(img_bytes).decode()

        # Dynamically discover currently-available free vision models
        models_to_try = _openrouter_free_vision_models(api_key)
        if not models_to_try:
            print("[OpenRouter] no free vision models available right now", file=sys.stderr)
            return None

        for model in models_to_try:
            payload = json.dumps({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                "temperature": 0.1,
                "max_tokens": 2048,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://meta-ads-static-checker-1.onrender.com",
                    "X-Title": "Meta Ads Checker",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                # OpenRouter free models can return an error inside 200 response
                if "error" in data:
                    print(f"[OpenRouter] {model} error: {data['error']}", file=sys.stderr)
                    continue
                print(f"[OpenRouter] success with model: {model}", file=sys.stderr)
                raw = data["choices"][0]["message"]["content"].strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                return json.loads(raw)
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                print(f"[OpenRouter] {model} HTTP {e.code}: {body}", file=sys.stderr)
                continue
            except json.JSONDecodeError as je:
                print(f"[OpenRouter] {model} JSON parse error: {je}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"[OpenRouter] {model} error: {type(e).__name__}: {e}", file=sys.stderr)
                continue

        print("[OpenRouter] all models failed", file=sys.stderr)
        return None

    except Exception as e:
        print(f"[OpenRouter] setup error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def analyze_ad(image_path: str, ratio_name: str | None, ad_copy: str = "") -> dict:
    gemini = _gemini_analyze(image_path, ratio_name, ad_copy)
    if gemini:
        return gemini

    openrouter = _openrouter_analyze(image_path, ratio_name)
    if openrouter:
        return openrouter

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
        "policy_compliance": _text_policy_check(ad_copy),
        "best_practices": {
            "has_brand_logo": None,
            "shows_product_in_use": None,
            "text_is_readable": text_pct > 3,
            "image_is_high_quality": sharpness >= 80,
            "shows_real_people": None,
            "color_appeal": "Strong" if saturation >= 50 else "Moderate" if saturation >= 30 else "Weak",
            "message_clarity": "Clear" if text_pct > 3 and sharpness >= 60 else "Unclear",
            "recommendations": [
                "Add Gemini API key for full AI-powered best practices assessment.",
                "Ensure your brand logo is visible for credibility on Meta feeds.",
                "Show people using your product in realistic settings to drive engagement.",
            ],
        },
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
