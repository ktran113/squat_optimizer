const form = document.getElementById("uploadForm");
const resultEl = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault(); // stop normal form submit

    const fileInput = document.getElementById("squat_video");
    if (!fileInput.files.length) {
    alert("Please select a video first.");
    return;
    }

    const formData = new FormData();
    // "file" must match the parameter name in your FastAPI endpoint:
    // async def analyze_squat_endpoint(file: UploadFile = File(...), fps: int = 30)
    formData.append("file", fileInput.files[0]);

    // Optional: you can add fps as a query param in the URL
    const apiUrl = "http://localhost:8000/analyze-video?fps=30";

    resultEl.textContent = "Uploading and analyzing...";

    try {
    const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        resultEl.textContent = `Error: ${response.status}\n${errorText}`;
        return;
    }

    const data = await response.json();
    resultEl.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
    resultEl.textContent = "Request failed: " + err.message;
    }
});