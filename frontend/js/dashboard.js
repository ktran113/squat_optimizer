const API_URL = "https://squat-optimizer.duckdns.org";

//check log in
const token = localStorage.getItem('token');
const userId = localStorage.getItem('userId');
const userName = localStorage.getItem('userName');

if (!token || !userId) {
    window.location.href = 'login.html';
}
document.getElementById('user-name').textContent = `Welcome, ${userName}`;

const uploadArea = document.getElementById('upload-area');
const uploadContent = document.getElementById('upload-content');
const videoInput = document.getElementById('video-input');
const fileNameDisplay = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');
const errorMessage = document.getElementById('error-message');
let lastResults = null;   //kept so the charts can be redrawn on resize
const resultsSection = document.getElementById('results-section');
const sessionsList = document.getElementById('sessions-list');
const videoPreviewWrap = document.getElementById('video-preview-wrap');
const videoPreview = document.getElementById('video-preview');
const videoScanOverlay = document.getElementById('video-scan-overlay');

let selectedFile = null;

//logout
document.getElementById('logout-btn').addEventListener('click', function() {
    localStorage.clear();
    window.location.href = 'index.html';
});

//drag drop
uploadArea.addEventListener('dragover', function(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', function(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', function(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});
//input change
videoInput.addEventListener('change', function(e) {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    const allowedTypes = ['.mp4', '.avi', '.mov', '.mkv'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();

    if (!allowedTypes.includes(ext)) {
        showError('Invalid file format. Please use MP4, AVI, MOV, or MKV.');
        return;
    }
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    analyzeBtn.disabled = false;
    hideError();

    // Show video preview
    const videoURL = URL.createObjectURL(file);
    videoPreview.src = videoURL;
    videoPreviewWrap.style.display = 'block';
    videoPreview.play();

    // hide dropzone when file in
    uploadArea.style.display = 'none';
}

analyzeBtn.addEventListener('click', async function() {
    if (!selectedFile) return;

    //prep form data, the backend reads fps from the video itself
    const formData = new FormData();
    formData.append('file', selectedFile);

    //show scanning overlay on video
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    hideError();
    videoScanOverlay.style.display = 'flex';
    videoPreview.play();

    try {
        const response = await fetch(`${API_URL}/analyze-video`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
            loadSessions(); // refresh session history
        } else {
            showError(data.detail || 'Analysis failed. Please try again.');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Cannot connect to server. Make sure backend is running!');
    } finally {
        videoScanOverlay.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

//canvas helpers ------------------------------------------------------------

const CHART_COLORS = {
    accent: '#00e5cc',
    amber: '#ffb84d',
    grid: '#2a2a40',
    dim: '#55556a',
    text: '#8888a0'
};

function prepCanvas(canvas, cssHeight) {
    //size the backing store to the device pixel ratio so lines stay crisp
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.parentElement.clientWidth - 36;
    canvas.style.height = cssHeight + 'px';
    canvas.width = Math.max(1, Math.round(cssWidth * ratio));
    canvas.height = Math.round(cssHeight * ratio);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    return { ctx, w: cssWidth, h: cssHeight };
}

function drawBarPath(data) {
    const canvas = document.getElementById('bar-path-chart');
    const empty = document.getElementById('bar-path-empty');
    const note = document.getElementById('bar-path-note');

    //nulls are frames where the detector found no bar
    const pts = (data.bar_path || []).filter(p => p && p[0] !== null && p[1] !== null);
    if (pts.length < 2) {
        canvas.style.display = 'none';
        empty.style.display = 'block';
        note.textContent = '';
        return;
    }
    canvas.style.display = 'block';
    empty.style.display = 'none';

    const { ctx, w, h } = prepCanvas(canvas, 220);
    const path = data.bar_path || [];

    //One panel per rep. Overlaying every rep in a single frame is unreadable
    //because each rep retraces the same vertical range.
    const bottoms = (data.reps || []).map(r => r.bottom_frame);
    const spans = bottoms.length
        ? bottoms.map((b, i) => {
            const prev = i > 0 ? Math.round((bottoms[i - 1] + b) / 2) : Math.max(0, 2 * b - path.length);
            const next = i < bottoms.length - 1 ? Math.round((b + bottoms[i + 1]) / 2) : path.length;
            return [Math.max(0, prev), Math.min(path.length, next)];
        })
        : [[0, path.length]];

    //shared scale across panels so reps are directly comparable
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const xMid = (Math.min(...xs) + Math.max(...xs)) / 2;
    const halfX = Math.max((Math.max(...xs) - Math.min(...xs)) / 2, 4);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const ySpan = Math.max(yMax - yMin, 1e-6);

    note.textContent = `${(Math.max(...xs) - Math.min(...xs)).toFixed(0)}px spread`
        + (bottoms.length ? ` · per rep` : '');

    const padY = 16, labelH = 14;
    const colW = w / spans.length;

    spans.forEach(([from, to], i) => {
        const cx = colW * i + colW / 2;
        const usable = colW * 0.38;

        //centre each rep on its own mean x, which is the value bar_path_dev
        //takes the standard deviation about, so the picture matches the number
        const own = [];
        for (let f = from; f < to; f++) {
            const p = path[f];
            if (p && p[0] !== null) own.push(p[0]);
        }
        const mean = own.length ? own.reduce((a, b) => a + b, 0) / own.length : xMid;

        const px = v => cx + ((v - mean) / halfX) * usable;
        const py = v => padY + ((v - yMin) / ySpan) * (h - padY * 2 - labelH);

        //plumb line: a perfectly vertical bar path
        ctx.strokeStyle = CHART_COLORS.grid;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, padY);
        ctx.lineTo(cx, h - padY - labelH);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.strokeStyle = CHART_COLORS.amber;
        ctx.lineWidth = 1.6;
        ctx.lineJoin = 'round';
        ctx.beginPath();
        let started = false;
        for (let f = from; f < to; f++) {
            const p = path[f];
            if (!p || p[0] === null || p[1] === null) { started = false; continue; }
            started ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1]));
            started = true;
        }
        ctx.stroke();

        const bottom = path[bottoms[i]];
        if (bottom && bottom[0] !== null) {
            ctx.fillStyle = CHART_COLORS.accent;
            ctx.beginPath();
            ctx.arc(px(bottom[0]), py(bottom[1]), 3.5, 0, Math.PI * 2);
            ctx.fill();
        }

        if (bottoms.length) {
            ctx.fillStyle = CHART_COLORS.text;
            ctx.font = '10px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`Rep ${i + 1}`, cx, h - 3);
        }
    });
}

function drawDepthChart(data) {
    const canvas = document.getElementById('depth-chart');
    const series = (data.depth_over_time || []).map(v => (v === null ? NaN : v));
    const finite = series.filter(v => Number.isFinite(v));
    if (finite.length < 2) return;

    const { ctx, w, h } = prepCanvas(canvas, 220);
    const pad = 18;
    const lo = Math.min(...finite), hi = Math.max(...finite);
    const span = Math.max(hi - lo, 1e-6);

    const px = i => pad + (i / Math.max(series.length - 1, 1)) * (w - pad * 2);
    //knee-hip shrinks as the lifter descends, so invert to make a squat read as a dip
    const py = v => pad + (1 - (v - lo) / span) * (h - pad * 2);

    ctx.strokeStyle = CHART_COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad, h - pad);
    ctx.lineTo(w - pad, h - pad);
    ctx.stroke();

    ctx.strokeStyle = CHART_COLORS.accent;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    let started = false;
    series.forEach((v, i) => {
        if (!Number.isFinite(v)) { started = false; return; }
        started ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v));
        started = true;
    });
    ctx.stroke();

    (data.reps || []).forEach(rep => {
        const f = rep.bottom_frame;
        if (!Number.isFinite(series[f])) return;
        ctx.fillStyle = CHART_COLORS.amber;
        ctx.beginPath();
        ctx.arc(px(f), py(series[f]), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = CHART_COLORS.text;
        ctx.font = '10px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(rep.rep_count, px(f), py(series[f]) + 16);
    });
}

function drawCharts(data) {
    drawBarPath(data);
    drawDepthChart(data);
}

//canvas does not reflow on its own, so redraw once resizing settles
let resizeTimer = null;
window.addEventListener('resize', function () {
    if (!lastResults) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => drawCharts(lastResults), 150);
});

function displayResults(data) {
    resultsSection.style.display = 'block';

    //summary
    document.getElementById('total-reps').textContent = data.total_reps || '-';

    const minAngle = data.reps && data.reps.length > 0
        ? Math.min(...data.reps.map(r => r.bottom_angle)).toFixed(1) + '\u00B0'
        : '-';
    document.getElementById('min-angle').textContent = minAngle;

    const avgTempo = data.tempo_per_rep && data.tempo_per_rep.length > 0
        ? (data.tempo_per_rep.reduce((a, b) => a + b, 0) / data.tempo_per_rep.length).toFixed(2) + 's'
        : '-';
    document.getElementById('avg-tempo').textContent = avgTempo;

    const avgBarDev = data.bar_path_dev && data.bar_path_dev.length > 0
        ? (data.bar_path_dev.reduce((a, b) => a + b, 0) / data.bar_path_dev.length).toFixed(1) + 'px'
        : '-';
    document.getElementById('bar-dev').textContent = avgBarDev;

    //charts, drawn after the section is visible so the canvas has a real width
    drawCharts(data);
    lastResults = data;

    //feedback
    document.getElementById('ai-feedback').textContent = data.ai_feedback || 'No feedback available.';

    // breakdown
    const repsList = document.getElementById('reps-list');
    repsList.innerHTML = '';

    if (data.reps && data.reps.length > 0) {
        data.reps.forEach((rep, index) => {
            const tempo = data.tempo_per_rep[index] ? data.tempo_per_rep[index].toFixed(2) + 's' : '-';
            const barDev = data.bar_path_dev[index] ? data.bar_path_dev[index].toFixed(1) + 'px' : '-';

            const repCard = document.createElement('div');
            repCard.className = 'rep-card';
            repCard.innerHTML = `
                <h4>Rep ${rep.rep_count}</h4>
                <p><strong>Depth:</strong> ${rep.depth}</p>
                <p><strong>Knee Angle:</strong> ${rep.bottom_angle.toFixed(1)}\u00B0</p>
                <p><strong>Tempo:</strong> ${tempo}</p>
                <p><strong>Bar Dev:</strong> ${barDev}</p>
            `;
            repsList.appendChild(repCard);
        });
    }
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

function hideError() {
    errorMessage.style.display = 'none';
}

//load history
async function loadSessions() {
    try {
        const response = await fetch(`${API_URL}/users/${userId}/sessions?limit=10`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.status === 401) {
            // Token expired
            localStorage.clear();
            window.location.href = 'login.html';
            return;
        }

        const sessions = await response.json();

        if (sessions.length === 0) {
            sessionsList.innerHTML = '<p class="empty-state">No workout sessions yet. Upload a video to get started!</p>';
            return;
        }

        sessionsList.innerHTML = '';
        sessions.forEach(session => {
            const date = new Date(session.created_at).toLocaleDateString();
            const sessionCard = document.createElement('div');
            sessionCard.className = 'session-card';
            sessionCard.innerHTML = `
                <div class="session-left">
                    <span class="session-date">${date}</span>
                    <div class="session-stats">
                        <span>Depth: ${session.avg_depth ? session.avg_depth.toFixed(1) + '\u00B0' : '-'}</span>
                        <span>Min Angle: ${session.min_knee_angle ? session.min_knee_angle.toFixed(1) + '\u00B0' : '-'}</span>
                        <span>Tempo: ${session.tempo ? session.tempo.toFixed(2) + 's' : '-'}</span>
                    </div>
                    ${session.ai_feedback ? `<p class="session-feedback">${session.ai_feedback.substring(0, 120)}...</p>` : ''}
                </div>
                <span class="session-reps-badge">${session.total_reps} reps</span>
            `;
            sessionCard.addEventListener('click', () => viewSession(session.id));
            sessionsList.appendChild(sessionCard);
        });

    } catch (error) {
        console.error('Error loading sessions:', error);
        sessionsList.innerHTML = '<p class="empty-state" style="color: var(--red);">Failed to load sessions.</p>';
    }
}

async function viewSession(sessionId) {
    try {
        const response = await fetch(`${API_URL}/sessions/${sessionId}`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (response.ok) {
            const session = response.json();
            //possibly expand session detals in a modal
            console.log('Session details:', session);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

//load sessions on page load
loadSessions();
