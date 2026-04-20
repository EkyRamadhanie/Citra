import os
import uuid

import cv2
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.secret_key = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def process_image(input_path: str, output_path: str, mode: str) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    if mode == "grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif mode == "binary":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("Unsupported processing mode.")

    success = cv2.imwrite(output_path, result)
    if not success:
        raise ValueError("Failed to save processed image.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/tugas1", methods=["GET", "POST"])
def tugas1():
    original_image = None
    processed_image = None
    selected_mode = "grayscale"

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "grayscale")

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Silakan pilih file gambar terlebih dahulu.", "danger")
            return redirect(url_for("tugas1"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau WEBP.", "danger")
            return redirect(url_for("tugas1"))

        safe_name = secure_filename(uploaded_file.filename)
        base_name, ext = os.path.splitext(safe_name)
        unique_token = uuid.uuid4().hex[:8]

        original_filename = f"{base_name}_{unique_token}{ext.lower()}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        uploaded_file.save(original_path)

        output_suffix = "gray" if selected_mode == "grayscale" else "binary"
        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_image(original_path, processed_path, selected_mode)
        except ValueError:
            if os.path.exists(original_path):
                os.remove(original_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas1"))
        except Exception:
            flash("Terjadi kesalahan saat memproses gambar.", "danger")
            return redirect(url_for("tugas1"))

        original_image = f"uploads/{original_filename}"
        processed_image = f"uploads/{processed_filename}"
        flash("Gambar berhasil diproses.", "success")

    return render_template(
        "tugas1.html",
        original_image=original_image,
        processed_image=processed_image,
        selected_mode=selected_mode,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash("Ukuran file terlalu besar. Maksimal 10 MB.", "danger")
    return redirect(url_for("tugas1"))


if __name__ == "__main__":
    app.run(debug=True)
