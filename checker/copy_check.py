import re

HEADLINE_MAX   = 40
PRIMARY_MAX    = 125

ENGAGEMENT_BAIT = [
    r"\blike if\b", r"\bshare (this|if)\b", r"\bcomment below\b",
    r"\btag (a |your )?(friend|someone|anyone)\b", r"\bvote (if|for)\b",
    r"\blike this (if|to)\b", r"\brepost (if|this)\b",
]

PERSONAL_ATTR = [
    r"\byour (race|ethnicity|religion|faith|belief|health|disability|age|gender|sexuality|income)\b",
    r"\b(are you|if you (are|have|suffer))\b.{0,30}(diabetic|obese|depressed|anxious|sick|poor|struggling)\b",
    r"\bpeople like you\b",
]

PROHIBITED_PATTERNS = [
    r"\b(guaranteed|100%)\s+(results?|weight loss|cure|income|profit|returns?)\b",
    r"\bmake money (fast|easy|online|from home)\b",
    r"\bget rich\b",
    r"\bbefore\s+and\s+after\b",
    r"\b(lose|lost)\s+\d+\s*(lbs?|kg|pounds?)\b",
    r"\bcasino\b", r"\bgamble\b", r"\bslots?\b",
    r"\b(buy|get)\s+(guns?|weapons?|ammo|ammunition)\b",
    r"\b(payday loan|cash advance)\b",
]


def _flag(text, patterns, label):
    text_lower = text.lower()
    hits = []
    for pattern in patterns:
        if re.search(pattern, text_lower):
            hits.append(pattern)
    return hits


def check_copy(headline: str, primary_text: str):
    issues  = []
    info    = []
    passed  = True

    headline     = (headline     or "").strip()
    primary_text = (primary_text or "").strip()

    # ── Headline checks ──
    if headline:
        h_len = len(headline)
        if h_len > HEADLINE_MAX:
            issues.append(
                f"Headline is {h_len} characters — Meta recommends under {HEADLINE_MAX} "
                "for full display on mobile. It will be truncated."
            )
            passed = False
        else:
            info.append(f"Headline length OK: {h_len}/{HEADLINE_MAX} characters.")

    # ── Primary text checks ──
    if primary_text:
        p_len = len(primary_text)
        if p_len > PRIMARY_MAX:
            issues.append(
                f"Primary text is {p_len} characters — Meta recommends under {PRIMARY_MAX} "
                "to avoid truncation in feed."
            )
            passed = False
        else:
            info.append(f"Primary text length OK: {p_len}/{PRIMARY_MAX} characters.")

    # ── Combined text for policy scans ──
    combined = f"{headline} {primary_text}".strip()

    if combined:
        bait_hits = _flag(combined, ENGAGEMENT_BAIT, "engagement bait")
        if bait_hits:
            issues.append(
                "Engagement bait language detected (e.g. 'like if', 'tag a friend', 'share this'). "
                "Meta policy prohibits artificially boosting engagement."
            )
            passed = False
        else:
            info.append("No engagement bait language detected.")

        attr_hits = _flag(combined, PERSONAL_ATTR, "personal attributes")
        if attr_hits:
            issues.append(
                "Personal attribute targeting language detected (references to race, health, religion, etc.). "
                "Meta policy prohibits implying knowledge of a user's personal attributes."
            )
            passed = False
        else:
            info.append("No personal attribute language detected.")

        prohibited_hits = _flag(combined, PROHIBITED_PATTERNS, "prohibited content")
        if prohibited_hits:
            issues.append(
                "Potentially prohibited content detected — guaranteed results, before/after claims, "
                "weight-loss numbers, or restricted product references. Review Meta's Advertising Standards."
            )
            passed = False
        else:
            info.append("No prohibited content patterns detected in copy.")

    return {
        "passed":        passed,
        "headline_len":  len(headline),
        "primary_len":   len(primary_text),
        "issues":        issues,
        "info":          info,
        "headline_max":  HEADLINE_MAX,
        "primary_max":   PRIMARY_MAX,
    }
