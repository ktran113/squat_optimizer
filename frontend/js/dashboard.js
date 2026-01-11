const API_URL = 'http://localhost:8000';

//check log in 
const token = localStorage.getItem('token');
const userId = localStorage.getItem('userId');
const userName = localStorage.getItem('userName');

if (!token || !userId) {
    window.location.href = 'login.html';
}
document.getElementById('user-name').textContent = `Welcome, ${userName}`;

const uploadArea = document.getElementById('upload-area');
const videoInput = document.getElementById('video-input');
const fileNameDisplay = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');
const fpsInput = document.getElementById('fps');
const analysisLoading = document.getElementById('analysis-loading');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');
const sessionsList = document.getElementById('sessions-list');

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
}

analyzeBtn.addEventListener('click', async function() {
    if (!selectedFile) return;

    const fps = parseInt(fpsInput.value);
    if (fps < 1 || fps > 240) {
        showError('FPS must be between 1 and 240');
        return;
    }
    //prep form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('fps', fps);

    //loading
    analysisLoading.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsSection.style.display = 'none';
    hideError();

    try {
        const response = await fetch(`${API_URL}/analyze-video?fps=${fps}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`
            },
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
            loadSessions(); // Refresh session history
        } else {
            showError(data.detail || 'Analysis failed. Please try again.');
        }

    } catch (error) {
        console.error('Error:', error);
        showError('Cannot connect to server. Make sure backend is running!');
    } finally {
        analysisLoading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    resultsSection.style.display = 'block';

    // Summary stats
    document.getElementById('total-reps').textContent = data.total_reps || '-';

    const minAngle = data.reps && data.reps.length > 0
        ? Math.min(...data.reps.map(r => r.bottom_angle)).toFixed(1) + '°'
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
                <p><strong>Knee Angle:</strong> ${rep.bottom_angle.toFixed(1)}°</p>
                <p><strong>Tempo:</strong> ${tempo}</p>
                <p><strong>Bar Deviation:</strong> ${barDev}</p>
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
            sessionsList.innerHTML = '<p class="no-sessions">No workout sessions yet. Upload a video to get started!</p>';
            return;
        }

        sessionsList.innerHTML = '';
        sessions.forEach(session => {
            const date = new Date(session.created_at).toLocaleDateString();
            const sessionCard = document.createElement('div');
            sessionCard.className = 'session-card';
            sessionCard.innerHTML = `
                <div class="session-header">
                    <span class="session-date">${date}</span>
                    <span class="session-reps">${session.total_reps} reps</span>
                </div>
                <div class="session-stats">
                    <span>Avg Depth: ${session.avg_depth ? session.avg_depth.toFixed(1) + '°' : '-'}</span>
                    <span>Min Angle: ${session.min_knee_angle ? session.min_knee_angle.toFixed(1) + '°' : '-'}</span>
                    <span>Tempo: ${session.tempo ? session.tempo.toFixed(2) + 's' : '-'}</span>
                </div>
                ${session.ai_feedback ? `<p class="session-feedback">${session.ai_feedback.substring(0, 150)}...</p>` : ''}
            `;
            sessionCard.addEventListener('click', () => viewSession(session.id));
            sessionsList.appendChild(sessionCard);
        });

    } catch (error) {
        console.error('Error loading sessions:', error);
        sessionsList.innerHTML = '<p class="error-text">Failed to load sessions.</p>';
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
            // Could expand to show full session details in a modal
            console.log('Session details:', session);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

//load sessions on page load
loadSessions();
