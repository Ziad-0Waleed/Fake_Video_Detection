const API = 'http://localhost:8000/api/v1';

// Health check
async function checkHealth() {
  const pill     = document.getElementById('api-pill');
  const pillText = document.getElementById('api-pill-text');
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    if (data.model_loaded) {
      pill.classList.add('online');
      pillText.innerHTML = 'Model Online';
      loadModelInfo();
    } else {
      pill.classList.add('offline');
      pillText.innerHTML = 'Model Not Loaded';
    }
  } catch {
    pill.classList.add('offline');
    pillText.innerHTML = 'API Offline';
  }
}

async function loadModelInfo() {
  try {
    const res  = await fetch(`${API}/model/info`);
    const data = await res.json();

    // Header pill
    document.getElementById('model-pill').style.display = 'flex';
    document.getElementById('model-pill-text').innerHTML = data.model_type;

    // Sidebar card
    document.getElementById('model-info-card').classList.add('visible');
    document.getElementById('mi-type').innerHTML   = data.model_type;
    document.getElementById('mi-thresh').innerHTML = data.threshold;
    document.getElementById('mi-status').innerHTML = 'Connected &#10003;';
  } catch { /* silent */ }
}

// File handling
const fileInput   = document.getElementById('file-input');
const dropZone    = document.getElementById('drop-zone');
const previewWrap = document.getElementById('preview-wrap');
const videoEl     = document.getElementById('video-preview');
const analyseBtn  = document.getElementById('analyse-btn');

let selectedFile = null;
let currentVideoUrl = null; // Track URL for memory management

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault(); dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('video/')) setFile(f);
});

function setFile(file) {
  selectedFile = file;

  // FREE MEMORY: Revoke the old URL before creating a new one
  if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);

  currentVideoUrl = URL.createObjectURL(file);
  videoEl.src = currentVideoUrl;

  document.getElementById('file-name-label').innerHTML = file.name;
  dropZone.style.display   = 'none';
  previewWrap.classList.add('visible');
  analyseBtn.disabled = false;

  // hide previous results
  document.getElementById('result-card').classList.remove('visible');
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('log-wrap').classList.remove('visible');
}

function clearFile() {
  selectedFile = null;

  // FREE MEMORY
  if (currentVideoUrl) URL.revokeObjectURL(currentVideoUrl);
  currentVideoUrl = null;

  videoEl.src  = '';
  fileInput.value = '';
  dropZone.style.display = '';
  previewWrap.classList.remove('visible');
  analyseBtn.disabled = true;
  document.getElementById('result-card').classList.remove('visible');
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('log-wrap').classList.remove('visible');
}

// Analyse
let logTimeouts = []; // Array to track active timeouts

async function analyse() {
  if (!selectedFile) return;

  const btn    = document.getElementById('analyse-btn');
  const errEl  = document.getElementById('error-msg');
  const logWrap = document.getElementById('log-wrap');
  const logText = document.getElementById('log-text');

  errEl.style.display = 'none';
  logWrap.classList.add('visible');
  btn.classList.add('loading');
  btn.disabled = true;

  // Clear any previous stray timeouts
  logTimeouts.forEach(clearTimeout);

  const steps = [
    { delay: 0,    cls: 'run', msg: 'Uploading video to server&hellip;' },
    { delay: 800,  cls: 'run', msg: 'Extracting up to 20 frames&hellip;' },
    { delay: 1800, cls: 'run', msg: 'Running EfficientNetB4 feature extraction&hellip;' },
    { delay: 3200, cls: 'run', msg: 'Running GRU sequence model&hellip;' },
  ];

  // Save timeout IDs so we can cancel them if the fetch finishes early
  logTimeouts = steps.map(s => setTimeout(() => {
    logText.innerHTML = `<span class="${s.cls}">${s.msg}</span>`;
  }, s.delay));

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch(`${API}/predict`, { method: 'POST', body: formData });

    // HALT THE FAKE LOGS
    logTimeouts.forEach(clearTimeout);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    logText.innerHTML = `<span class="ok">&#10003; Analysis complete</span>`;
    renderResult(data);
  } catch (e) {
    logTimeouts.forEach(clearTimeout); // Stop logs on error too
    logText.innerHTML = `<span class="err">&times; ${e.message}</span>`;
    errEl.style.display = 'block';
    errEl.innerHTML   = `&#9888;  ${e.message}`;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// Render result
function renderResult(data) {
  const isFake  = data.label === 'FAKE';
  const fakePct = (data.fake_probability * 100).toFixed(1);
  const realPct = (data.real_probability * 100).toFixed(1);

  // Verdict
  const vEl = document.getElementById('verdict');
  vEl.innerHTML = isFake ? '&#9888; FAKE' : '&#10003; REAL';
  vEl.className   = `verdict ${isFake ? 'fake' : 'real'}`;

  document.getElementById('verdict-sub').innerHTML = isFake
    ? 'This video shows signs of synthetic manipulation.'
    : 'No deepfake artefacts detected in this video.';

  // Gauge — shows fake probability
  document.getElementById('gauge-pct').innerHTML = `${fakePct}%`;
  const fill = document.getElementById('gauge-fill');
  fill.className = `gauge-fill${isFake ? ' danger' : ''}`;
  setTimeout(() => fill.style.width = `${fakePct}%`, 50);

  // Stats
  document.getElementById('s-fake').innerHTML   = `${fakePct}%`;
  document.getElementById('s-fake').className     = `stat-val ${isFake ? 'red' : ''}`;
  document.getElementById('s-real').innerHTML   = `${realPct}%`;
  document.getElementById('s-real').className     = `stat-val ${!isFake ? 'green' : ''}`;
  document.getElementById('s-conf').innerHTML   = data.confidence;
  document.getElementById('s-frames').innerHTML = data.frames_analyzed;

  document.getElementById('result-card').classList.add('visible');
  document.getElementById('result-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

checkHealth();