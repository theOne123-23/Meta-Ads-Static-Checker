import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from checker.size_check import check_size
from checker.safe_zone_check import check_safe_zone
from checker.ai_analysis import analyze_ad

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB for bulk

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/check-bulk", methods=["POST"])
def check_bulk():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    results = []
    for file in files:
        if not file or file.filename == "":
            continue
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": "File type not allowed. Use PNG, JPG, GIF, or WEBP.",
            })
            continue

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            size_result = check_size(save_path)
            ratio = size_result.get("detected_ratio")

            safe_zone_result = check_safe_zone(save_path, ratio) if ratio else None

            try:
                ai_result = analyze_ad(save_path, ratio)
            except Exception as e:
                ai_result = {"error": f"AI analysis failed: {str(e)}"}

            results.append({
                "filename": file.filename,
                "size": size_result,
                "safe_zone": safe_zone_result,
                "ai": ai_result,
            })
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)
