import anthropic
import base64
import json
import io
from PIL import Image

SAFE_ZONE_MARGINS = {
    "9:16":   {"top": "14%", "bottom": "22%", "left": "5%", "right": "5%"},
    "1.91:1": {"top": "10%", "bottom": "10%", "left": "5%", "right": "5%"},
}


def _encode_image(image_path: str) -> tuple[str, str]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1200, 1200), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, "image/jpeg"


def analyze_ad(image_path: str, ratio_name: str | None) -> dict:
    is_safe_zone_ratio = ratio_name in SAFE_ZONE_MARGINS
    margins = SAFE_ZONE_MARGINS.get(ratio_name or "", {})

    if is_safe_zone_ratio:
        safe_zone_instructions = f"""
SAFE ZONE ANALYSIS (required for {ratio_name}):
- Danger zones: Top {margins['top']}, Bottom {margins['bottom']}, Left/Right {margins['left']} of image edges
- The SAFE ZONE is the inner rectangle — fully visible to users
- Danger zones are covered by platform UI (profile name, CTA buttons, navigation icons)

IMPORTANT RULES for safe zone judgment:
1. Only flag foreground elements that carry INFORMATION: text, headlines, logos, product shots, price tags, CTA buttons, badges
2. A foreground element must be CLEARLY and SIGNIFICANTLY cut off or placed in the danger zone to FAIL — not just touching the edge
3. Add a margin of error: if text/content is close to but not clearly INTO the danger zone, call it PASS with a note
4. ALWAYS PASS if all key text, logos and CTAs are visibly centered or in the middle portion of the image
5. Background photos, textures, gradients, colors extending to edges = always IGNORE, never flag
6. Red borders or decorative frames around the image = always IGNORE
"""
        safe_zone_json = '''"safe_zone": {
    "verdict": "PASS or FAIL — only FAIL if important content is CLEARLY cut into danger zones",
    "elements_in_danger_zone": ["list specific text/logo/CTA elements clearly in danger zones — empty list if none or if only background"],
    "background_extends_to_edge": true or false,
    "explanation": "One clear sentence. If PASS: confirm content is safely centered. If FAIL: name exactly what element is cut off and where."
  },'''
    else:
        safe_zone_instructions = """
SAFE ZONE: This is a 1:1 ratio ad. The ENTIRE image is fully visible in the Facebook/Instagram feed — no safe zone restrictions apply. Skip safe zone analysis and return PASS automatically.
"""
        safe_zone_json = '''"safe_zone": {
    "verdict": "PASS",
    "elements_in_danger_zone": [],
    "background_extends_to_edge": false,
    "explanation": "1:1 ratio ads are fully visible in feed — no safe zone restrictions apply."
  },'''

    prompt = f"""You are an expert Meta (Facebook/Instagram) advertising AI analyst. You analyze static ad creatives and provide detailed, specific, actionable feedback like a senior performance marketing consultant.

Aspect ratio: {ratio_name or 'unknown'}
{safe_zone_instructions}

CREATIVE QUALITY ANALYSIS — evaluate these Andromeda ranking factors:
- Thumb-stop power: Bold color, unexpected composition, prominent face, pattern interrupt?
- Text clarity: Is copy readable on a 4-inch mobile screen? Font size appropriate?
- Value proposition: Is the offer/benefit visible within 0.5 seconds?
- Contrast: Strong subject-background separation?
- Composition: Clear focal point, not cluttered?
- Social proof: Stars, numbers, testimonials, trust badges?
- CTA: Is the call-to-action clear and compelling?
- Brand: Logo/brand colors present and consistent?

PERFORMANCE PREDICTION — based on all factors above, predict how this ad will perform on Meta placements.

Return ONLY valid JSON (absolutely no text outside the JSON):
{{
  {safe_zone_json}
  "quality": {{
    "score": 0-100,
    "thumb_stop_power": "Strong / Moderate / Weak",
    "text_clarity": "Clear / Borderline / Poor",
    "value_prop_visible": true or false,
    "contrast": "High / Medium / Low",
    "clutter_level": "Clean / Moderate / Cluttered"
  }},
  "performance_score": 0-100,
  "winning_verdict": "Winning Ad / Promising / Needs Work / Not Ready",
  "is_winning_ad": true or false,
  "strengths": [
    "Specific strength 1 — be concrete, e.g. 'Strong 70% OFF offer is immediately visible'",
    "Specific strength 2"
  ],
  "issues": [
    "Specific issue 1 — be concrete, e.g. 'Logo in top-left is too small to read on mobile'",
    "Specific issue 2 (if any)"
  ],
  "improvements": [
    "Actionable tip 1 — specific and detailed, e.g. 'Move the BOOK NOW button 20% higher to keep it away from the bottom UI bar'",
    "Actionable tip 2 — specific and detailed",
    "Actionable tip 3 — specific and detailed"
  ],
  "analysis": "Write 3 sentences as an expert AI analyst: (1) What this ad does well overall, (2) The biggest risk or weakness, (3) One key recommendation to improve performance. Be specific about THIS ad's content, not generic advice."
}}"""

    img_data, media_type = _encode_image(image_path)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1400,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return json.loads(raw)
