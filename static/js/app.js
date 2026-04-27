document.addEventListener("DOMContentLoaded", () => {
  setupStaggerAnimation();
  setupUploadInteractions();
  setupModeHint();
  setupCompareSlider();
  setupSubmitLoading();
});

function setupStaggerAnimation() {
  const animated = Array.from(document.querySelectorAll("[data-stagger]"));
  if (!animated.length) return;

  animated.forEach((el, index) => {
    el.style.animationDelay = `${Math.min(index * 80, 360)}ms`;
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  animated.forEach((el) => observer.observe(el));
}

function setupUploadInteractions() {
  const dropZone = document.querySelector("[data-dropzone]");
  const fileInput = document.getElementById("image");
  const fileName = document.getElementById("fileName");
  const previewWrap = document.getElementById("previewWrap");
  const previewImage = document.getElementById("previewImage");

  if (!dropZone || !fileInput) return;

  const updateFileState = () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      if (fileName) {
        fileName.textContent = "Belum ada file dipilih.";
      }
      if (previewWrap && previewImage) {
        previewWrap.hidden = true;
        previewImage.src = "";
      }
      return;
    }

    if (fileName) {
      const maxLength = 45;
      fileName.textContent =
        file.name.length > maxLength ? `${file.name.slice(0, maxLength - 3)}...` : file.name;
    }

    if (previewWrap && previewImage && file.type.startsWith("image/")) {
      const objectUrl = URL.createObjectURL(file);
      previewImage.src = objectUrl;
      previewWrap.hidden = false;
      previewImage.onload = () => URL.revokeObjectURL(objectUrl);
    }
  };

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", updateFileState);

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("is-dragover");
    });
  });

  ["dragleave", "dragend", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("is-dragover");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    const droppedFiles = event.dataTransfer ? event.dataTransfer.files : null;
    if (!droppedFiles || !droppedFiles.length) return;

    fileInput.files = droppedFiles;
    updateFileState();
  });
}

function setupModeHint() {
  const modeHint = document.getElementById("modeHint");
  const modeInputs = document.querySelectorAll("input[name='mode']");
  if (!modeHint || !modeInputs.length) return;

  const defaultMessages = {
    grayscale: "Mode grayscale cocok untuk menonjolkan struktur dan tekstur citra tanpa dipengaruhi warna.",
    binary: "Mode binary cocok untuk segmentasi sederhana dengan pemisahan objek dan latar belakang secara tegas.",
    canny: "Canny memanfaatkan dua ambang untuk mendeteksi tepi kuat dan tepi lemah secara lebih stabil.",
    sobel: "Sobel menghitung gradien horizontal dan vertikal untuk menonjolkan perubahan intensitas citra.",
    roberts: "Roberts memakai kernel 2x2 sehingga respons tepi diagonal terlihat lebih tajam.",
    robets: "Roberts memakai kernel 2x2 sehingga respons tepi diagonal terlihat lebih tajam.",
    prewitt: "Prewitt menggunakan operator gradien sederhana untuk menampilkan kontur objek dengan cepat.",
  };

  const updateHint = () => {
    const selected = document.querySelector("input[name='mode']:checked");
    if (!selected) {
      modeHint.textContent = "Pilih metode pemrosesan untuk melihat penjelasan singkat.";
      return;
    }

    const inputHint = selected.dataset.hint;
    if (inputHint) {
      modeHint.textContent = inputHint;
      return;
    }

    modeHint.textContent = defaultMessages[selected.value] || defaultMessages.grayscale;
  };

  modeInputs.forEach((input) => input.addEventListener("change", updateHint));
  updateHint();
}

function setupCompareSlider() {
  const compareRange = document.getElementById("compareRange");
  const overlay = document.getElementById("compareProcessed");
  const comparePercent = document.getElementById("comparePercent");
  if (!compareRange || !overlay || !comparePercent) return;

  const updateCompare = () => {
    const value = Number(compareRange.value);
    overlay.style.width = `${value}%`;
    comparePercent.textContent = String(value);
  };

  compareRange.addEventListener("input", updateCompare);
  updateCompare();
}

function setupSubmitLoading() {
  const form = document.getElementById("processForm");
  const processBtn = document.getElementById("processBtn");
  if (!form || !processBtn) return;

  form.addEventListener("submit", () => {
    processBtn.classList.add("is-loading");
    processBtn.setAttribute("disabled", "disabled");
  });
}
