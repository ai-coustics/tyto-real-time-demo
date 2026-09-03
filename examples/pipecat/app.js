// Tyto Pipecat demo, browser client.
//
// Thin client: capture the mic and play the agent over a WebRTC peer connection
// to the Pipecat backend, and render the UI from messages the backend sends over
// the WebRTC data channel. All scoring, the three adaptation layers, and the keys
// live on the backend. The metric rendering here is identical to the websocket
// web demo (examples/web/app.js); only the transport differs: WebRTC media for
// audio (so the browser's echo cancellation and jitter buffer do the work) and a
// data channel for the score/layer/transcript/log messages.

const $ = (id) => document.getElementById(id);
const $status = $("status"), $mic = $("mic"), $log = $("log"), $banner = $("banner");
const $userTx = $("user-tx"), $agentTx = $("agent-tx");

const SERIES_HISTORY_MS = 30000;
const emaAlpha = 0.3;            // sparkline smoothing only

// Tyto 1.1 dimensions, in the docs' display order. 1.1 merged the old
// media_speech into interfering_speech and added codec_degradation.
const ENV_KEYS = ["noise", "speaker_reverb", "speaker_loudness", "interfering_speech", "packet_loss", "codec_degradation"];
const LABELS = {
  noise: "Noise", speaker_reverb: "Speaker Reverb", speaker_loudness: "Speaker Loudness",
  interfering_speech: "Interfering Speech", packet_loss: "Packet Loss",
  codec_degradation: "Codec Degradation",
};
const ICONS = {
  noise: '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>',
  speaker_reverb: '<circle cx="12" cy="12" r="2"/><path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.2 19.1 19.1"/>',
  speaker_loudness: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>',
  interfering_speech: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
  codec_degradation: '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>',
  packet_loss: '<line x1="2" y1="2" x2="22" y2="22"/><path d="M8.5 16.5a5 5 0 0 1 7 0"/><path d="M2 8.82a15 15 0 0 1 4.17-2.65"/><path d="M10.66 5c4.01-.36 8.14.9 11.34 3.76"/><path d="M16.85 11.25a10 10 0 0 1 2.22 1.68"/><path d="M5 13a10 10 0 0 1 5.24-2.76"/><line x1="12" y1="20" x2="12.01" y2="20"/>',
};
const icon = (k) => ICONS[k] ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[k]}</svg>` : "";
const NO_POLARITY = new Set(["speaker_loudness", "speaker_reverb"]);
const THRESHOLDS = {
  noise: [0.20, 0.45], interfering_speech: [0.15, 0.35], packet_loss: [0.05, 0.15],
  codec_degradation: [0.30, 0.50], speaker_reverb: [0.25, 0.55], speaker_loudness: [0.12, 0.25],
};
// Tyto Risk Score bands from the docs. Mirrors COMPOSITE_TH in decision.py.
const COMPOSITE_TH = [0.30, 0.50];
// Mirrors NUDGE_THRESHOLD_DEFAULT / _MIN / _MAX in decision.py.
const NUDGE_TH = { min: 0.30, max: 0.50, default: 0.31 };
const DESCRIPTIONS = {
  noise: "Ambient noise behind the speaker, relative to the speaker's level. High = the noise is loud compared to the speaker.",
  packet_loss: "Audio dropouts or discontinuities: packet loss, jitter, frame erasure, or CPU overload.",
  interfering_speech: "Competing speech behind the main speaker: people nearby, or a TV, radio or phone playing.",
  codec_degradation: "Compression artifacts from the codec carrying the audio: low bitrate, narrowband telephony, transcoding.",
  speaker_loudness: "Loudness level of the main speaker. Informational only.",
  speaker_reverb: "Low = dry, near-field audio; high = reverberant, far-field audio. Informational only.",
};

// ── UI rendering (identical to examples/web/app.js) ───────────────────────────
function setStatus(state, label) { $status.className = "badge " + state; $status.innerHTML = `<span class="dot"></span>${label}`; }
function setTytoState(state, text) {
  const el = $("tyto-state"); if (!el) return;
  el.className = "composite-formula" + (state ? " ts-" + state : "");
  el.textContent = text || "lower is better";
}
function colorFor(t) {
  if (t.startsWith("tyto.nudge") || t.startsWith("tool.")) return "t-orange";
  if (t.startsWith("tyto.aware")) return "t-purple";
  if (t.startsWith("tyto.vad")) return "t-blue";
  if (t.includes("transcript")) return "t-blue";
  if (t === "error" || t === "tyto.error") return "t-red";
  if (t.startsWith("tyto.")) return "t-green";
  return "t-gray";
}
function log(type, text) {
  const ts = new Date().toTimeString().slice(0, 8);
  const e = document.createElement("div"); e.className = "entry";
  e.innerHTML = `<span class="ts">${ts}</span><span class="type ${colorFor(type)}">${type}</span><span class="preview"></span>`;
  e.querySelector(".preview").textContent = text || "";
  $log.appendChild(e); $log.scrollTop = $log.scrollHeight;
}
function bucket(key, v) {
  if (NO_POLARITY.has(key)) return "green";
  const th = THRESHOLDS[key] || [0.30, 0.50];
  return v < th[0] ? "green" : v < th[1] ? "yellow" : "red";
}
function compositeBucket(v) { return v < COMPOSITE_TH[0] ? "green" : v < COMPOSITE_TH[1] ? "yellow" : "red"; }

const sparkCanvases = new Map();
function buildCards() {
  sparkCanvases.clear();
  const grid = $("env-grid"); grid.innerHTML = "";
  for (const k of ENV_KEYS) {
    const c = document.createElement("div"); c.className = "card"; c.title = DESCRIPTIONS[k] || "";
    c.innerHTML = `<div class="c-label">${icon(k)}<span>${LABELS[k]}</span></div>` +
      `<div class="c-bar"><div class="c-fill" id="fill-${k}"></div></div>` +
      `<div class="c-val" id="val-${k}">–</div>` +
      `<div class="spark-wrap"><canvas class="spark" id="spark-${k}"></canvas></div>` +
      `<div class="c-desc">${DESCRIPTIONS[k] || ""}</div>`;
    grid.appendChild(c);
    sparkCanvases.set(k, c.querySelector(`#spark-${k}`));
  }
}
function renderComposite(c) {
  if (c == null) return;
  const fill = $("composite-fill"), val = $("composite-val");
  fill.style.width = Math.round(Math.min(1, c) * 100) + "%";
  fill.className = "composite-fill " + compositeBucket(c);
  val.textContent = c.toFixed(2);
}
function renderMetrics(m) {
  for (const k of ENV_KEYS) {
    const v = m[k]; const fill = $(`fill-${k}`), val = $(`val-${k}`);
    if (v == null || !fill) continue;
    fill.style.width = Math.round(Math.min(1, v) * 100) + "%";
    fill.className = "c-fill " + bucket(k, v);
    val.textContent = v.toFixed(2);
  }
}
function setVoiceFocus(available, on) {
  const box = $("vf-toggle"), note = $("vf-note");
  if (!box) return;
  box.disabled = !available;
  box.checked = !!on;
  box.parentElement.classList.toggle("unavailable", !available);
  // One short line each: the row is a fixed height and the note does not wrap.
  note.textContent = !available
    ? "enhancement model did not load"
    : on
      ? "Quail VF 2.2 is cleaning the agent's input"
      : "the agent hears your raw microphone";
}

// transcripts
let userFinals = [], userInterim = "", agentFinals = [], agentInterim = "";
function renderTx() {
  const draw = (el, f, i) => { el.innerHTML = ""; for (const x of f) { const d = document.createElement("div"); d.className = "line"; d.textContent = x; el.appendChild(d); } if (i) { const d = document.createElement("div"); d.className = "line interim"; d.textContent = i; el.appendChild(d); } el.scrollTop = el.scrollHeight; };
  draw($userTx, userFinals, userInterim); draw($agentTx, agentFinals, agentInterim);
}

// sparklines
class MetricSeries {
  constructor(keys) { this.keys = keys; this.points = new Map(); this.emaPrev = new Map(); for (const k of keys) this.points.set(k, []); }
  push(t, values, alpha) {
    const a = Math.min(1, Math.max(0, alpha));
    for (const key of this.keys) {
      const value = values[key];
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      const prev = this.emaPrev.get(key);
      const ema = prev === undefined ? value : a * value + (1 - a) * prev;
      this.emaPrev.set(key, ema);
      this.points.get(key).push({ t, raw: value, ema });
    }
  }
  prune(now, historyMs) { const cutoff = now - historyMs - 2000; for (const arr of this.points.values()) { let i = 0; while (i < arr.length && arr[i].t < cutoff) i++; if (i > 0) arr.splice(0, i); } }
  get(key) { return this.points.get(key) || []; }
  clear() { for (const k of this.keys) this.points.set(k, []); this.emaPrev.clear(); }
}
const metricSeries = new MetricSeries(ENV_KEYS);
let rafId = null;
function syncSparkCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return { width, height };
}
function strokeSeries(ctx, points, pick, width, height, now, color, alpha) {
  const start = now - SERIES_HISTORY_MS; let started = false;
  ctx.beginPath();
  for (const p of points) {
    const x = ((p.t - start) / SERIES_HISTORY_MS) * width;
    const y = height - (Math.max(0, Math.min(1, pick(p))) * height);
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  if (!started) return;
  ctx.strokeStyle = color; ctx.globalAlpha = alpha; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke(); ctx.globalAlpha = 1;
}
function renderSeries() {
  const now = Date.now();
  metricSeries.prune(now, SERIES_HISTORY_MS);
  for (const key of ENV_KEYS) {
    const canvas = sparkCanvases.get(key); if (!canvas) continue;
    const { width, height } = syncSparkCanvas(canvas);
    const ctx = canvas.getContext("2d"); if (!ctx) continue;
    ctx.clearRect(0, 0, width, height);
    const points = metricSeries.get(key); if (!points.length) continue;
    strokeSeries(ctx, points, (p) => p.raw, width, height, now, "#6993FF", 0.95);
    strokeSeries(ctx, points, (p) => p.ema, width, height, now, "#F9F9F9", 0.85);
    const latest = points[points.length - 1];
    const markerY = height - (Math.max(0, Math.min(1, latest.raw)) * height);
    ctx.fillStyle = bucket(key, latest.raw) === "green" ? "#00BFA6" : bucket(key, latest.raw) === "yellow" ? "#F4B942" : "#E28C7C";
    ctx.beginPath(); ctx.arc(width - 4, markerY, 2.5, 0, Math.PI * 2); ctx.fill();
  }
  rafId = requestAnimationFrame(renderSeries);
}
function startSeriesLoop() { if (rafId != null) cancelAnimationFrame(rafId); rafId = requestAnimationFrame(renderSeries); }
function clearSeries() { metricSeries.clear(); for (const c of sparkCanvases.values()) { const ctx = c.getContext("2d"); if (ctx) ctx.clearRect(0, 0, c.width, c.height); } }

// nudge sensitivity slider (tells the backend over the data channel)
function setNudgeThreshold(v) {
  const val = Math.min(NUDGE_TH.max, Math.max(NUDGE_TH.min, v));
  $("nudge-th").value = val.toFixed(2);
  $("nudge-th-val").textContent = `≥ ${val.toFixed(2)}`;
  $("composite-marker").style.left = (val * 100).toFixed(1) + "%";
  send({ type: "nudge_threshold", value: val });
}

// ── Transport: WebRTC peer connection + data channel ──────────────────────────
let pc = null, dc = null, pcId = null, micStream = null, audioEl = null, connected = false;

function send(obj) { if (dc && dc.readyState === "open") dc.send(JSON.stringify(obj)); }

function waitForIceGathering(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => { if (peer.iceGatheringState === "complete") { peer.removeEventListener("icegatheringstatechange", check); resolve(); } };
    peer.addEventListener("icegatheringstatechange", check);
    setTimeout(resolve, 3000); // safety net: don't hang if a candidate stalls
  });
}

async function negotiate(restart) {
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitForIceGathering(pc);
  const resp = await fetch("/api/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, pc_id: pcId, restart_pc: !!restart }),
  });
  const answer = await resp.json();
  pcId = answer.pc_id || pcId;
  await pc.setRemoteDescription(answer);
}

async function start() {
  setStatus("connecting", "Connecting…"); buildCards();
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    });

    pc = new RTCPeerConnection({ iceServers: [] });
    audioEl = new Audio(); audioEl.autoplay = true;
    pc.ontrack = (e) => { audioEl.srcObject = e.streams[0]; };
    pc.onconnectionstatechange = () => {
      log("rtc.state", pc.connectionState);
      // Drive the badge from the real connection state so it never gets stuck on
      // "Connecting…"; the backend's "status"/"tyto_state" messages refine it.
      if (pc.connectionState === "connected") setStatus("live", "Live");
      else if (["failed", "closed", "disconnected"].includes(pc.connectionState)) stop();
    };

    for (const track of micStream.getAudioTracks()) pc.addTrack(track, micStream);

    // The client creates the data channel; the backend listens for it and uses
    // it for both UI messages and WebRTC signalling (renegotiation, peer-left).
    dc = pc.createDataChannel("tyto");
    dc.onopen = () => {
      log("rtc.datachannel", "open");
      // The bootstrap call ran with dc === null, so the server never heard
      // it. Re-send now, or the marker and the real trip point disagree.
      send({ type: "nudge_threshold", value: +$("nudge-th").value });
    };
    dc.onmessage = onMessage;

    await negotiate(false);

    connected = true; $mic.classList.add("live");
    log("rtc.open", "connected to backend");
  } catch (err) {
    setStatus("error", "Error"); log("error", String(err)); stop();
  }
}

function stop() {
  if (!connected && !pc) return;
  connected = false; $mic.classList.remove("live");
  if (dc) { try { dc.close(); } catch {} dc = null; }
  if (pc) { try { pc.close(); } catch {} pc = null; }
  if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
  if (audioEl) { try { audioEl.pause(); } catch {} audioEl.srcObject = null; audioEl = null; }
  pcId = null;
  clearSeries(); setTytoState(null);
  userFinals = []; agentFinals = []; userInterim = ""; agentInterim = ""; renderTx();
  setStatus("", "Disconnected");
}

async function handleSignalling(message) {
  if (!message || !pc) return;
  if (message.type === "renegotiate") {
    try { await negotiate(false); } catch (err) { log("error", "renegotiate: " + err); }
  } else if (message.type === "peerLeft") {
    stop();
  }
}

function onMessage(ev) {
  let m;
  try { m = JSON.parse(ev.data); } catch { return; }
  if (m.type === "signalling") { handleSignalling(m.message); return; }
  switch (m.type) {
    case "status": setStatus(m.state, m.label); break;
    case "tyto_state":
      if (m.state === "loading") setTytoState("loading", "loading Tyto model…");
      else if (m.state === "warming") setTytoState("warming", "warming up - keep talking");
      else if (m.state === "live") setTytoState(null);
      else if (m.state === "error") setTytoState("error", m.text || "Tyto error");
      log(`tyto.${m.state}`, m.text || ""); break;
    case "scores":
      renderMetrics(m.scores); renderComposite(m.scores.risk_score);
      metricSeries.push(Date.now(), m.scores, emaAlpha);
      break;
    case "transcript":
      // Interim text is a whole-utterance snapshot, not a delta: Deepgram Flux
      // re-sends the full transcript on every update, so append would repeat it.
      if (m.who === "user") { if (m.final) { if (m.text) userFinals.push(m.text); userInterim = ""; } else userInterim = m.text; }
      else { if (m.final) { if (m.text) agentFinals.push(m.text); agentInterim = ""; } else agentInterim = m.text; }
      renderTx(); break;
    case "nudge":
      $banner.textContent = `Tyto: ${m.label} = ${m.value.toFixed(2)}, nudging the agent`;
      $banner.classList.add("visible"); setTimeout(() => $banner.classList.remove("visible"), 6000);
      log("tyto.nudge", m.text); break;
    case "voice_focus": setVoiceFocus(m.available, m.on); break;
    case "log": log(m.kind, m.text); break;
  }
}

$("nudge-th").addEventListener("input", (e) => setNudgeThreshold(+e.target.value));
$("vf-toggle").addEventListener("change", (e) => {
  // Optimistic: the server answers with the state it actually reached, which
  // is off if the model did not load.
  send({ type: "voice_focus", value: e.target.checked });
});
$mic.addEventListener("click", () => (connected ? stop() : start()));
buildCards();
setNudgeThreshold(NUDGE_TH.default);
startSeriesLoop();
