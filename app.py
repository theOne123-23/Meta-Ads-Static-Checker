import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from checker.size_check import check_size
from checker.safe_zone_check import check_safe_zone

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30MB max

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

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        size_result = check_size(save_path)

        safe_zone_result = None
        if size_result.get("detected_ratio"):
            safe_zone_result = check_safe_zone(save_path, size_result["detected_ratio"])

        result = {**size_result, "safe_zone_check": safe_zone_result}
    finally:
        os.remove(save_path)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
