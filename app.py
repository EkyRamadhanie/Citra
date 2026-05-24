import os
import uuid

import cv2
import numpy as np
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
MORPH_MODE_SUFFIX = {
    "dilation": "dilation",
    "erosion": "erosion",
}
MORPH2_MODE_SUFFIX = {
    "boundary": "boundary",
    "convex_hull": "convex_hull",
    "skeletonizing": "skeletonizing",
}
SEGMENTATION_MODE_SUFFIX = {
    "hsv": "hsv_segmented",
    "kmeans": "kmeans_segmented",
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


def process_morphology(input_path: str, output_paths: dict[str, str], mode: str) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if mode == "dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif mode == "erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    else:
        raise ValueError("Unsupported morphology mode.")

    step_images = {
        "grayscale": grayscale,
        "binary": binary,
        "processed": result,
    }

    for key, step_image in step_images.items():
        success = cv2.imwrite(output_paths[key], step_image)
        if not success:
            raise ValueError("Failed to save processed image.")


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    skeleton = np.zeros(binary.shape, np.uint8)
    working_image = binary.copy()
    structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(working_image, structuring_element)
        opened = cv2.dilate(eroded, structuring_element)
        residual = cv2.subtract(working_image, opened)
        skeleton = cv2.bitwise_or(skeleton, residual)
        working_image = eroded.copy()

        if cv2.countNonZero(working_image) == 0:
            break

    return skeleton


def process_morphology_2(input_path: str, output_paths: dict[str, str], mode: str) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    if mode == "boundary":
        eroded = cv2.erode(binary, kernel, iterations=1)
        result = cv2.subtract(binary, eroded)
    elif mode == "convex_hull":
        result = np.zeros_like(binary)
        contours, _hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) <= 0:
                continue
            hull = cv2.convexHull(contour)
            cv2.drawContours(result, [hull], -1, 255, thickness=cv2.FILLED)
    elif mode == "skeletonizing":
        result = skeletonize_binary(binary)
    else:
        raise ValueError("Unsupported morphology mode.")

    step_images = {
        "grayscale": grayscale,
        "binary": binary,
        "processed": result,
    }

    for key, step_image in step_images.items():
        success = cv2.imwrite(output_paths[key], step_image)
        if not success:
            raise ValueError("Failed to save processed image.")


def process_segmentation(input_path: str, output_path: str, mode: str, cluster_count: int = 3) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    smoothed = cv2.GaussianBlur(image, (5, 5), 0)

    if mode == "hsv":
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        saliency_mask = (saturation > 40) & (value > 40)
        if np.any(saliency_mask):
            hue_values = hue[saliency_mask]
            dominant_hue = int(np.bincount(hue_values, minlength=180).argmax())
            hue_margin = 15
            lower_bound = np.array([max(0, dominant_hue - hue_margin), 40, 40], dtype=np.uint8)
            upper_bound = np.array([min(179, dominant_hue + hue_margin), 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        result = cv2.bitwise_and(image, image, mask=mask)
    elif mode == "kmeans":
        pixels = smoothed.reshape((-1, 3)).astype(np.float32)
        cluster_count = max(2, min(8, int(cluster_count)))
        if pixels.shape[0] < cluster_count:
            raise ValueError("Uploaded image is too small for clustering.")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _compactness, labels, centers = cv2.kmeans(
            pixels,
            cluster_count,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS,
        )

        centers = centers.astype(np.uint8)
        segmented = centers[labels.flatten()].reshape(image.shape)
        result = segmented
    else:
        raise ValueError("Unsupported segmentation mode.")

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


@app.route("/tugas3", methods=["GET", "POST"])
def tugas3():
    original_image = None
    grayscale_image = None
    binary_image = None
    processed_image = None
    selected_mode = "dilation"

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "dilation").lower()

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Silakan pilih file gambar terlebih dahulu.", "danger")
            return redirect(url_for("tugas3"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau WEBP.", "danger")
            return redirect(url_for("tugas3"))

        safe_name = secure_filename(uploaded_file.filename)
        base_name, ext = os.path.splitext(safe_name)
        unique_token = uuid.uuid4().hex[:8]

        original_filename = f"{base_name}_{unique_token}{ext.lower()}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        uploaded_file.save(original_path)

        output_suffix = MORPH_MODE_SUFFIX.get(selected_mode)
        if output_suffix is None:
            cleanup_files(original_path)
            flash("Metode morfologi tidak didukung.", "danger")
            return redirect(url_for("tugas3"))

        grayscale_filename = f"{base_name}_{unique_token}_gray.png"
        binary_filename = f"{base_name}_{unique_token}_binary.png"
        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"

        grayscale_path = os.path.join(app.config["UPLOAD_FOLDER"], grayscale_filename)
        binary_path = os.path.join(app.config["UPLOAD_FOLDER"], binary_filename)
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_morphology(
                original_path,
                {
                    "grayscale": grayscale_path,
                    "binary": binary_path,
                    "processed": processed_path,
                },
                selected_mode,
            )
        except ValueError:
            cleanup_files(original_path, grayscale_path, binary_path, processed_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas3"))
        except Exception:
            cleanup_files(original_path, grayscale_path, binary_path, processed_path)
            flash("Terjadi kesalahan saat memproses gambar.", "danger")
            return redirect(url_for("tugas3"))

        original_image = f"uploads/{original_filename}"
        grayscale_image = f"uploads/{grayscale_filename}"
        binary_image = f"uploads/{binary_filename}"
        processed_image = f"uploads/{processed_filename}"
        flash("Gambar berhasil diproses bertahap.", "success")

    return render_template(
        "tugas3.html",
        original_image=original_image,
        grayscale_image=grayscale_image,
        binary_image=binary_image,
        processed_image=processed_image,
        selected_mode=selected_mode,
    )


@app.route("/tugas4", methods=["GET", "POST"])
def tugas4():
    original_image = None
    grayscale_image = None
    binary_image = None
    processed_image = None
    selected_mode = "boundary"

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "boundary").lower()

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Silakan pilih file gambar terlebih dahulu.", "danger")
            return redirect(url_for("tugas4"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau WEBP.", "danger")
            return redirect(url_for("tugas4"))

        safe_name = secure_filename(uploaded_file.filename)
        base_name, ext = os.path.splitext(safe_name)
        unique_token = uuid.uuid4().hex[:8]

        original_filename = f"{base_name}_{unique_token}{ext.lower()}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        uploaded_file.save(original_path)

        output_suffix = MORPH2_MODE_SUFFIX.get(selected_mode)
        if output_suffix is None:
            cleanup_files(original_path)
            flash("Metode morfologi 2 tidak didukung.", "danger")
            return redirect(url_for("tugas4"))

        grayscale_filename = f"{base_name}_{unique_token}_gray.png"
        binary_filename = f"{base_name}_{unique_token}_binary.png"
        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"

        grayscale_path = os.path.join(app.config["UPLOAD_FOLDER"], grayscale_filename)
        binary_path = os.path.join(app.config["UPLOAD_FOLDER"], binary_filename)
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_morphology_2(
                original_path,
                {
                    "grayscale": grayscale_path,
                    "binary": binary_path,
                    "processed": processed_path,
                },
                selected_mode,
            )
        except ValueError:
            cleanup_files(original_path, grayscale_path, binary_path, processed_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas4"))
        except Exception:
            cleanup_files(original_path, grayscale_path, binary_path, processed_path)
            flash("Terjadi kesalahan saat memproses gambar.", "danger")
            return redirect(url_for("tugas4"))

        original_image = f"uploads/{original_filename}"
        grayscale_image = f"uploads/{grayscale_filename}"
        binary_image = f"uploads/{binary_filename}"
        processed_image = f"uploads/{processed_filename}"
        flash("Gambar berhasil diproses bertahap.", "success")

    return render_template(
        "tugas4.html",
        original_image=original_image,
        grayscale_image=grayscale_image,
        binary_image=binary_image,
        processed_image=processed_image,
        selected_mode=selected_mode,
    )


@app.route("/tugas5", methods=["GET", "POST"])
def tugas5():
    original_image = None
    processed_image = None
    selected_mode = "hsv"
    cluster_count = 3

    if request.method == "POST":
        uploaded_file = request.files.get("image")
        selected_mode = request.form.get("mode", "hsv").lower()
        if selected_mode == "kmeans":
            try:
                cluster_count = int(request.form.get("cluster_count", "3"))
            except (TypeError, ValueError):
                cluster_count = 3
            cluster_count = max(2, min(8, cluster_count))

        if uploaded_file is None or uploaded_file.filename == "":
            flash("Silakan pilih file gambar terlebih dahulu.", "danger")
            return redirect(url_for("tugas5"))

        if not allowed_file(uploaded_file.filename):
            flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, BMP, atau WEBP.", "danger")
            return redirect(url_for("tugas5"))

        safe_name = secure_filename(uploaded_file.filename)
        base_name, ext = os.path.splitext(safe_name)
        unique_token = uuid.uuid4().hex[:8]

        original_filename = f"{base_name}_{unique_token}{ext.lower()}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_filename)
        uploaded_file.save(original_path)

        output_suffix = SEGMENTATION_MODE_SUFFIX.get(selected_mode)
        if output_suffix is None:
            cleanup_files(original_path)
            flash("Metode segmentasi tidak didukung.", "danger")
            return redirect(url_for("tugas5"))

        processed_filename = f"{base_name}_{unique_token}_{output_suffix}.png"
        processed_path = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)

        try:
            process_segmentation(original_path, processed_path, selected_mode, cluster_count)
        except ValueError:
            cleanup_files(original_path, processed_path)
            flash("File tidak valid atau gagal diproses.", "danger")
            return redirect(url_for("tugas5"))
        except Exception:
            cleanup_files(original_path, processed_path)
            flash("Terjadi kesalahan saat memproses gambar.", "danger")
            return redirect(url_for("tugas5"))

        original_image = f"uploads/{original_filename}"
        processed_image = f"uploads/{processed_filename}"
        flash("Gambar berhasil disegmentasi.", "success")

    return render_template(
        "tugas5.html",
        original_image=original_image,
        processed_image=processed_image,
        selected_mode=selected_mode,
        cluster_count=cluster_count,
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(_error):
    flash("Ukuran file terlalu besar. Maksimal 10 MB.", "danger")

    if request.path.startswith("/tugas5"):
        return redirect(url_for("tugas5"))
    if request.path.startswith("/tugas4"):
        return redirect(url_for("tugas4"))
    if request.path.startswith("/tugas3"):
        return redirect(url_for("tugas3"))
    if request.path.startswith("/tugas2"):
        return redirect(url_for("tugas2"))
    if request.path.startswith("/tugas1"):
        return redirect(url_for("tugas1"))
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
