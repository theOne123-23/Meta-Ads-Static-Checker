import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from checker.size_check import check_size
from checker.safe_zone_check import check_safe_zone
from checker.quality_check import check_quality
from checker.copy_check import check_copy

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check", methods=["POST"])
def check():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use PNG, JPG, GIF, or WEBP."}), 400

    headline     = request.form.get("headline", "").strip()
    primary_text = request.form.get("primary_text", "").strip()

    filename  = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        size_result    = check_size(save_path)
        quality_result = check_quality(save_path)

        safe_zone_result = None
        if size_result.get("detected_ratio"):
            safe_zone_result = check_safe_zone(save_path, size_result["detected_ratio"])

        copy_result = None
        if headline or primary_text:
            copy_result = check_copy(headline, primary_text)

        # Andromeda score (0-100)
        score = _andromeda_score(size_result, quality_result, safe_zone_result, copy_result)

        result = {
            **size_result,
            "safe_zone_check": safe_zone_result,
            "quality_check":   quality_result,
            "copy_check":      copy_result,
            "andromeda_score": score,
        }
    finally:
        os.remove(save_path)

    return jsonify(result)


def _andromeda_score(size, quality, safe_zone, copy):
    score = 0

    # Size/ratio (25 pts)
    if size.get("passed"):
        score += 25

    # Image quality (35 pts)
    if quality:
        if quality["sharpness_score"] >= 80:
            score += 12
        elif quality["sharpness_score"] >= 40:
            score += 6
        if quality["contrast_score"] >= 45:
            score += 12
        elif quality["contrast_score"] >= 25:
            score += 6
        if quality["text_area_pct"] <= 20:
            score += 11
        elif quality["text_area_pct"] <= 30:
            score += 5

    # Safe zone (20 pts)
    if safe_zone:
        warnings = safe_zone.get("warnings", [])
        if len(warnings) == 0:
            score += 20
        elif len(warnings) == 1:
            score += 10

    # Copy (20 pts)
    if copy:
        if copy.get("passed"):
            score += 20
        else:
            issue_count = len(copy.get("issues", []))
            score += max(0, 20 - issue_count * 7)

    return min(score, 100)


if __name__ == "__main__":
    app.run(debug=True)
