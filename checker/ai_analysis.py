import anthropic
import base64
import json
import io
from PIL import Image

SAFE_ZONE_MARGINS = {
    "1:1":    {"top": "5%",  "bottom": "5%",  "left": "5%",  "right": "5%"},
    "9:16":   {"top": "14%", "bottom": "22%", "left": "5%",  "right": "5%"},
    "1.91:1": {"top": "10%", "bottom": "10%", "left": "5%",  "right": "5%"},
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
    margins = SAFE_ZONE_MARGINS.get(ratio_name or "", {})
    safe_desc = (
        f"Top {margins.get('top','~5%')} and bottom {margins.get('bottom','~5%')} "
        f"of the image are DANGER ZONES — they will be covered by platform UI "
        f"(profile name, CTA button, engagement icons). "
        f"Left and right {margins.get('left','~5%')} are also danger zones."
    ) if margins else "Safe zone: inner ~80% of the image is fully visible."

    prompt = f"""You are an expert Meta (Facebook/Instagram) advertising creative analyst trained on Andromeda ranking criteria.

The uploaded image is a static ad with aspect ratio: {ratio_name or 'unknown'}.

SAFE ZONE CONTEXT:
{safe_desc}
The safe zone is the inner rectangle that will be FULLY visible to users. Anything outside it gets covered by platform UI.

YOUR TASK — analyze the image and return a JSON object ONLY (no explanation text outside the JSON):

Step 1 — SAFE ZONE SCAN:
- Look at what foreground elements are present: text, headlines, logos, product images, price tags, buttons, badges, CTAs, faces, star ratings.
- Check whether any of these important elements appear to be positioned near or beyond the safe zone edges.
- CRITICAL EXCEPTION: Background images, textures, gradients, solid color fills, or blurred backgrounds that naturally extend to the image edges are COMPLETELY NORMAL and must NOT be flagged. Only flag elements that carry information.

Step 2 — CREATIVE QUALITY (Meta Andromeda factors):
- Thumb-stop power: Is there a bold color, unexpected composition, or prominent face?
- Text clarity: Is text large enough and readable on a 4-inch phone screen?
- Value proposition: Is the offer or benefit immediately visible (within 0.5 seconds)?
- Contrast: Is there strong contrast between subject and background?
- Clutter: Is the ad clean and focused on one message?
- Social proof: Any stars, numbers, testimonials?
- CTA clarity: Is the next action obvious?

Step 3 — WINNING AD ASSESSMENT:
Based on Meta's Creative Advantage and Andromeda standards, rate and assess this ad.

Return ONLY this JSON (fill every field, use realistic scores):
{{
  "safe_zone": {{
    "verdict": "PASS or FAIL",
    "elements_in_danger_zone": ["list any specific text/logo/CTA/product found near or outside safe zone edges — empty list if none"],
    "background_extends_to_edge": true or false,
    "explanation": "One sentence explaining the safe zone verdict"
  }},
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
  "strengths": ["strength 1", "strength 2"],
  "issues": ["specific issue 1", "specific issue 2"],
  "improvements": [
    "Specific actionable improvement 1",
    "Specific actionable improvement 2",
    "Specific actionable improvement 3"
  ],
  "analysis": "2-3 sentence holistic assessment of the ad and its predicted performance on Meta placements"
}}"""

    img_data, media_type = _encode_image(image_path)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
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

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return json.loads(raw)
