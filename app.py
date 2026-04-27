import os
import uuid

import cv2
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
TASK1_MODE_SUFFIX = {
    "grayscale": "gray",
    "binary": "binary",
}
EDGE_MODE_SUFFIX = {
    "canny": "canny",
    "sobel": "sobel",
    "roberts": "roberts",
    "robets": "roberts",
    "prewitt": "prewitt",
}

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


def cleanup_files(*file_paths: str) -> None:
    for path in file_paths:
        if path and os.path.exists(path):
            os.remove(path)


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


def process_edge_detection(input_path: str, output_path: str, mode: str) -> None:
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Uploaded file is not a valid image.")

    if mode == "canny":
        result = cv2.Canny(gray, 100, 200)
    elif mode == "sobel":
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif mode in {"roberts", "robets"}:
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)).astype("float64")
        kernel_x[0, 1] = 0
        kernel_x[1, 0] = 0
        kernel_x[1, 1] = -1
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)).astype("float64")
        kernel_y[0, 0] = 0
        kernel_y[1, 1] = 0
        kernel_y[1, 0] = -1

        grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        result = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif mode == "prewitt":
        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)).astype("float64")
        kernel_x[:, 1] = 0
        kernel_x[:, 2] = -1
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)).astype("float64")
        kernel_y[1, :] = 0
        kernel_y[2, :] = -1

        grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        result = cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    else:
        raise ValueError("Unsupported edge detection mode.")

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
        selected_mode = request.form.get("mode", "grayscale").lower()

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

        output_suffix = TASK1_MODE_SUFFIX.get(selected_mode)
        if output_suffix is None:
            cleanup_files(original_path)
            flash("Metode pengolahan tidak didukung.", "danger")
            return redirect(url_for("tugas1"))

        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_image(original_path, processed_path, selected_mode)
        except ValueError:
            cleanup_files(original_path, processed_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas1"))
        except Exception:
            cleanup_files(original_path, processed_path)
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


@app.route("/tugas2", methods=["GET", "POST"])
def tugas2():
    original_image = None
    processed_image = None
    selected_mode = "canny"

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "canny").lower()

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Silakan pilih file gambar terlebih dahulu.", "danger")
            return redirect(url_for("tugas2"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau WEBP.", "danger")
            return redirect(url_for("tugas2"))

        safe_name = secure_filename(uploaded_file.filename)
        base_name, ext = os.path.splitext(safe_name)
        unique_token = uuid.uuid4().hex[:8]

        original_filename = f"{base_name}_{unique_token}{ext.lower()}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        uploaded_file.save(original_path)

        output_suffix = EDGE_MODE_SUFFIX.get(selected_mode)
        if output_suffix is None:
            cleanup_files(original_path)
            flash("Metode deteksi tepi tidak didukung.", "danger")
            return redirect(url_for("tugas2"))

        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_edge_detection(original_path, processed_path, selected_mode)
        except ValueError:
            cleanup_files(original_path, processed_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas2"))
        except Exception:
            cleanup_files(original_path, processed_path)
            flash("Terjadi kesalahan saat memproses gambar.", "danger")
            return redirect(url_for("tugas2"))

        original_image = f"uploads/{original_filename}"
        processed_image = f"uploads/{processed_filename}"
        flash("Gambar berhasil diproses.", "success")

    return render_template(
        "tugas2.html",
        original_image=original_image,
        processed_image=processed_image,
        selected_mode=selected_mode,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash("Ukuran file terlalu besar. Maksimal 10 MB.", "danger")

    if request.path.startswith("/tugas2"):
        return redirect(url_for("tugas2"))
    if request.path.startswith("/tugas1"):
        return redirect(url_for("tugas1"))
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
