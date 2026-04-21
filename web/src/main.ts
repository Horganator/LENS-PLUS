import "./styles.css";
import { startCameraStream, stopStream } from "./media/cameraSource";
import type { DirectionalContext, InferenceMessage } from "./types/messages";
import { isInferenceMessage } from "./types/messages";
import { UiLogger } from "./ui/logger";
import { WebRtcClient } from "./webrtc/client";
import { getSignalingBaseUrl } from "./webrtc/signaling";

type PageName = "home" | "camera" | "admin";
type ThemeName = "dark" | "light";
type StatusTone = "up" | "warn" | "down" | "neutral";
type SensorPermissionState = "unknown" | "granted" | "denied" | "unsupported";

type RotationRateDps = {
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
};

type OrientationDegrees = {
  alpha: number | null;
  beta: number | null;
  gamma: number | null;
  absolute: boolean | null;
};

type DirectionalTelemetryMessage = {
  type: "client_sensor";
  sensor: "gyro";
  timestamp_ms: number;
  rotation_rate_dps: RotationRateDps | null;
  orientation_deg: OrientationDegrees | null;
};

const SEND_TARGET_FPS = 30;
const ADMIN_PREVIEW_POLL_MS = 750;
const THEME_STORAGE_KEY = "lensplus-theme";
const SENSOR_TELEMETRY_INTERVAL_MS = 120;
const RELATIVE_HEADING_CENTER_BAND_DEG = 3;
const RELATIVE_PITCH_CENTER_BAND_DEG = 3;
const HEADING_SMOOTHING_ALPHA = 0.18;
const PITCH_SMOOTHING_ALPHA = 0.22;
const HEADING_JITTER_DEADBAND_DEG = 1.5;
const PITCH_JITTER_DEADBAND_DEG = 1.5;

type DebugSession = {
  session_id: string;
  connection_state: string;
  ice_state: string;
  analysis_target_fps: number;
  total_frames: number;
  processed_frames: number;
  dropped_frames: number;
  incoming_fps: number;
  processed_fps: number;
  last_frame_at: string | null;
  latest_jpeg_at: string | null;
  has_snapshot: boolean;
  snapshot_errors: number;
  last_snapshot_error: string | null;
  directional_samples_received: number;
  directional_messages_ignored: number;
  directional_parse_errors: number;
  latest_directional_received_at: string | null;
  latest_directional_client_timestamp_ms: number | null;
  latest_processed_directional: DirectionalContext | null;
  updated_at: string;
};

type RawDebugSession = Partial<DebugSession> & { session_id: string };

function mustQuery<T extends Element>(selector: string): T {
  const node = document.querySelector<T>(selector);
  if (!node) {
    throw new Error(`Missing element: ${selector}`);
  }
  return node;
}

function loadStoredTheme(): ThemeName | null {
  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (storedTheme === "dark" || storedTheme === "light") {
      return storedTheme;
    }
    return null;
  } catch {
    return null;
  }
}

function saveStoredTheme(theme: ThemeName): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    return;
  }
}

function formatFps(value: number): string {
  if (!Number.isFinite(value) || value < 0) {
    return "0.0";
  }
  return value.toFixed(1);
}

function humanizeState(rawState: string): string {
  const normalized = rawState.trim();
  if (!normalized) {
    return "N/A";
  }

  return normalized
    .replace(/_/g, " ")
    .split(" ")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function toneForPeerState(rawState: string): StatusTone {
  const state = rawState.toLowerCase();
  if (state === "connected") {
    return "up";
  }
  if (state === "new" || state === "connecting" || state === "disconnected") {
    return "warn";
  }
  if (state === "failed") {
    return "down";
  }
  return "neutral";
}

function toneForIceState(rawState: string): StatusTone {
  const state = rawState.toLowerCase();
  if (state === "connected" || state === "completed") {
    return "up";
  }
  if (state === "new" || state === "checking" || state === "disconnected") {
    return "warn";
  }
  if (state === "failed") {
    return "down";
  }
  return "neutral";
}

function toneForHealth(rawStatus: string): StatusTone {
  if (rawStatus === "ok") {
    return "up";
  }
  if (rawStatus === "unreachable") {
    return "down";
  }
  return "warn";
}

function toneForDirectionalLink(
  directional: DirectionalContext | null,
  sampleCount: number
): StatusTone {
  if (sampleCount <= 0) {
    return "warn";
  }

  if (!directional) {
    return "warn";
  }

  if (directional.is_stale) {
    return "warn";
  }

  return "up";
}

function formatDirectionalLinkLabel(
  directional: DirectionalContext | null,
  sampleCount: number
): string {
  if (sampleCount <= 0) {
    return "No samples";
  }

  if (!directional) {
    return "Awaiting frame";
  }

  if (directional.is_stale) {
    return "Stale";
  }

  return "Fresh";
}

function formatDirectionalAge(direction: DirectionalContext | null): string {
  if (!direction || typeof direction.age_ms !== "number" || !Number.isFinite(direction.age_ms)) {
    return "n/a";
  }

  return `${Math.max(0, Math.round(direction.age_ms))} ms`;
}

function createStatusPill(label: string, tone: StatusTone): HTMLElement {
  const pill = document.createElement("span");
  pill.className = `status-pill tone-${tone}`;
  pill.textContent = label;
  return pill;
}

function setBadgeCell(target: HTMLElement, label: string, tone: StatusTone): void {
  target.innerHTML = "";
  target.classList.remove("value-mono");
  target.appendChild(createStatusPill(label, tone));
}

function setTextCell(target: HTMLElement, value: string, monospace = false): void {
  target.textContent = value;
  target.classList.toggle("value-mono", monospace);
}

function normalizeDebugSession(session: RawDebugSession): DebugSession {
  return {
    session_id: session.session_id,
    connection_state: session.connection_state ?? "unknown",
    ice_state: session.ice_state ?? "unknown",
    analysis_target_fps:
      typeof session.analysis_target_fps === "number" ? session.analysis_target_fps : 0,
    total_frames: typeof session.total_frames === "number" ? session.total_frames : 0,
    processed_frames:
      typeof session.processed_frames === "number" ? session.processed_frames : 0,
    dropped_frames: typeof session.dropped_frames === "number" ? session.dropped_frames : 0,
    incoming_fps: typeof session.incoming_fps === "number" ? session.incoming_fps : 0,
    processed_fps: typeof session.processed_fps === "number" ? session.processed_fps : 0,
    last_frame_at: session.last_frame_at ?? null,
    latest_jpeg_at: session.latest_jpeg_at ?? null,
    has_snapshot: Boolean(session.has_snapshot),
    snapshot_errors:
      typeof session.snapshot_errors === "number" ? session.snapshot_errors : 0,
    last_snapshot_error: session.last_snapshot_error ?? null,
    directional_samples_received:
      typeof session.directional_samples_received === "number"
        ? session.directional_samples_received
        : 0,
    directional_messages_ignored:
      typeof session.directional_messages_ignored === "number"
        ? session.directional_messages_ignored
        : 0,
    directional_parse_errors:
      typeof session.directional_parse_errors === "number"
        ? session.directional_parse_errors
        : 0,
    latest_directional_received_at: session.latest_directional_received_at ?? null,
    latest_directional_client_timestamp_ms:
      typeof session.latest_directional_client_timestamp_ms === "number"
        ? session.latest_directional_client_timestamp_ms
        : null,
    latest_processed_directional:
      typeof session.latest_processed_directional === "object" &&
      session.latest_processed_directional !== null
        ? session.latest_processed_directional
        : null,
    updated_at: session.updated_at ?? ""
  };
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("#app not found");
}

app.innerHTML = `
  <main class="app-shell">
    <div class="global-actions">
      <button id="theme-toggle" class="ghost theme-toggle" type="button">Light mode</button>
    </div>

    <section id="home-page" class="page home-page">
      <h1>LENS+ Control</h1>
      <p>Select where you want to work.</p>
      <div class="home-actions">
        <button id="go-camera" class="primary">Camera</button>
        <button id="go-admin">Admin Panel</button>
      </div>
    </section>

    <section id="camera-page" class="page hidden">
      <div class="page-top">
        <button id="camera-home" class="ghost">Home</button>
      </div>
      <div class="camera-wrap">
        <video id="camera-preview" autoplay muted playsinline></video>
      </div>
      <section class="direction-card" aria-live="polite">
        <div class="direction-card-header">
          <h2 class="direction-card-title">Direction</h2>
          <button id="direction-reset" class="ghost direction-reset" type="button">
            Recenter
          </button>
        </div>
        <div class="direction-card-body">
          <div class="direction-dial" aria-hidden="true">
            <span class="direction-origin"></span>
            <span id="direction-needle" class="direction-needle"></span>
            <span class="direction-center-dot"></span>
          </div>
          <div class="direction-readout">
            <p id="direction-relative-value" class="direction-value">--</p>
            <p id="direction-relative-text" class="direction-text">
              Start feed to calibrate origin.
            </p>
            <p id="direction-pitch-value" class="direction-subvalue">Pitch --</p>
            <p id="direction-pitch-arrow" class="direction-arrow">^v At origin</p>
            <p id="direction-pitch-text" class="direction-subtext">
              Keep device level to calibrate tilt.
            </p>
          </div>
        </div>
      </section>
      <button id="start-feed" class="primary">Start Feed</button>
    </section>

    <section id="admin-page" class="page hidden">
      <div class="page-top">
        <button id="admin-home" class="ghost">Home</button>
      </div>

      <section class="panel">
        <h2>Feeds</h2>
        <div class="feed-grid">
          <figure class="feed-box">
            <figcaption>Live Feed</figcaption>
            <div class="feed-stage">
              <img id="admin-live-preview" alt="Latest frame from selected session" hidden />
              <p id="admin-live-status" class="feed-status">No active session.</p>
            </div>
          </figure>
        </div>
      </section>

      <section class="panel">
        <div class="panel-title-row">
          <h2>System Status Dashboard</h2>
          <span id="connection-badge" class="status-badge is-neutral">Disconnected</span>
        </div>

        <table class="status-table">
          <tbody>
            <tr>
              <th>API Health</th>
              <td id="status-api"></td>
            </tr>
            <tr>
              <th>Peer State</th>
              <td id="status-peer"></td>
            </tr>
            <tr>
              <th>ICE State</th>
              <td id="status-ice"></td>
            </tr>
            <tr>
              <th>Session ID</th>
              <td id="status-session-id"></td>
            </tr>
            <tr>
              <th>Session Count</th>
              <td id="status-session-count"></td>
            </tr>
            <tr>
              <th>Send Target FPS</th>
              <td id="status-send-fps"></td>
            </tr>
            <tr>
              <th>Analysis Target FPS</th>
              <td id="status-analysis-fps"></td>
            </tr>
            <tr>
              <th>Incoming FPS</th>
              <td id="status-incoming-fps"></td>
            </tr>
            <tr>
              <th>Processed FPS</th>
              <td id="status-processed-fps"></td>
            </tr>
            <tr>
              <th>Dropped Frames</th>
              <td id="status-dropped-frames"></td>
            </tr>
            <tr>
              <th>Total Frames Received</th>
              <td id="status-total-frames"></td>
            </tr>
            <tr>
              <th>Directional Link</th>
              <td id="status-directional-link"></td>
            </tr>
            <tr>
              <th>Directional Age</th>
              <td id="status-directional-age"></td>
            </tr>
            <tr>
              <th>Sensor Samples</th>
              <td id="status-directional-samples"></td>
            </tr>
            <tr>
              <th>Sensor Parse Errors</th>
              <td id="status-directional-errors"></td>
            </tr>
            <tr>
              <th>Sensor Messages Ignored</th>
              <td id="status-directional-ignored"></td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="panel">
        <h2>Logs</h2>
        <ul id="event-log"></ul>
      </section>
    </section>
  </main>
`;

const homePageEl = mustQuery<HTMLElement>("#home-page");
const cameraPageEl = mustQuery<HTMLElement>("#camera-page");
const adminPageEl = mustQuery<HTMLElement>("#admin-page");
const themeToggleEl = mustQuery<HTMLButtonElement>("#theme-toggle");
const goCameraEl = mustQuery<HTMLButtonElement>("#go-camera");
const goAdminEl = mustQuery<HTMLButtonElement>("#go-admin");
const cameraHomeEl = mustQuery<HTMLButtonElement>("#camera-home");
const adminHomeEl = mustQuery<HTMLButtonElement>("#admin-home");
const startFeedEl = mustQuery<HTMLButtonElement>("#start-feed");
const cameraPreviewEl = mustQuery<HTMLVideoElement>("#camera-preview");
const directionResetEl = mustQuery<HTMLButtonElement>("#direction-reset");
const directionNeedleEl = mustQuery<HTMLElement>("#direction-needle");
const directionRelativeValueEl = mustQuery<HTMLElement>("#direction-relative-value");
const directionRelativeTextEl = mustQuery<HTMLElement>("#direction-relative-text");
const directionPitchValueEl = mustQuery<HTMLElement>("#direction-pitch-value");
const directionPitchArrowEl = mustQuery<HTMLElement>("#direction-pitch-arrow");
const directionPitchTextEl = mustQuery<HTMLElement>("#direction-pitch-text");
const adminLivePreviewEl = mustQuery<HTMLImageElement>("#admin-live-preview");
const adminLiveStatusEl = mustQuery<HTMLElement>("#admin-live-status");
const connectionBadgeEl = mustQuery<HTMLElement>("#connection-badge");
const statusApiEl = mustQuery<HTMLElement>("#status-api");
const statusPeerEl = mustQuery<HTMLElement>("#status-peer");
const statusIceEl = mustQuery<HTMLElement>("#status-ice");
const statusSessionIdEl = mustQuery<HTMLElement>("#status-session-id");
const statusSessionCountEl = mustQuery<HTMLElement>("#status-session-count");
const statusSendFpsEl = mustQuery<HTMLElement>("#status-send-fps");
const statusAnalysisFpsEl = mustQuery<HTMLElement>("#status-analysis-fps");
const statusIncomingFpsEl = mustQuery<HTMLElement>("#status-incoming-fps");
const statusProcessedFpsEl = mustQuery<HTMLElement>("#status-processed-fps");
const statusDroppedFramesEl = mustQuery<HTMLElement>("#status-dropped-frames");
const statusTotalFramesEl = mustQuery<HTMLElement>("#status-total-frames");
const statusDirectionalLinkEl = mustQuery<HTMLElement>("#status-directional-link");
const statusDirectionalAgeEl = mustQuery<HTMLElement>("#status-directional-age");
const statusDirectionalSamplesEl = mustQuery<HTMLElement>("#status-directional-samples");
const statusDirectionalErrorsEl = mustQuery<HTMLElement>("#status-directional-errors");
const statusDirectionalIgnoredEl = mustQuery<HTMLElement>("#status-directional-ignored");
const eventLogEl = mustQuery<HTMLUListElement>("#event-log");

let currentPage: PageName = "home";
let activeTheme: ThemeName = loadStoredTheme() ?? "dark";
let activeStream: MediaStream | null = null;
let activeSessionId: string | null = null;
let peerState: RTCPeerConnectionState = "closed";
const sendTargetFps = SEND_TARGET_FPS;
let latestHealthStatus = "unknown";
let latestSessions: DebugSession[] = [];
let latestSelectedSession: DebugSession | null = null;
let adminPreviewSessionId: string | null = null;
let adminPreviewHasSnapshot = false;
let viewportConstraintTimer: number | null = null;
let directionalPermissionsResolved = false;
let motionPermissionState: SensorPermissionState = "unknown";
let orientationPermissionState: SensorPermissionState = "unknown";
let directionalTelemetryActive = false;
let latestRotationRateDps: RotationRateDps | null = null;
let latestOrientationDegrees: OrientationDegrees | null = null;
let lastDirectionalTelemetrySentAtMs = 0;
let headingOriginAlphaDeg: number | null = null;
let headingCurrentAlphaDeg: number | null = null;
let headingSmoothedAlphaDeg: number | null = null;
let pitchOriginBetaDeg: number | null = null;
let pitchCurrentBetaDeg: number | null = null;
let pitchSmoothedBetaDeg: number | null = null;
let displayedRelativeHeadingDeg: number | null = null;
let displayedRelativePitchDeg: number | null = null;
const logger = new UiLogger(eventLogEl);
const apiBaseUrl = getSignalingBaseUrl();
const errorCache = new Map<string, string>();

showAdminLiveStatus("No active session.");

adminLivePreviewEl.addEventListener("load", () => {
  adminLivePreviewEl.hidden = false;
  adminLiveStatusEl.hidden = true;
});

adminLivePreviewEl.addEventListener("error", () => {
  if (!adminPreviewSessionId) {
    return;
  }
  adminLivePreviewEl.hidden = true;
  showAdminLiveStatus("Waiting for latest frame...");
});

const webrtc = new WebRtcClient(
  (state) => {
    peerState = state;
    renderConnectionBadge();
    renderLatestDashboard();
  },
  (payload) => {
    handleInferencePayload(payload);
  },
  (message) => {
    logger.log(message);
  }
);

goCameraEl.addEventListener("click", () => {
  showPage("camera");
});

goAdminEl.addEventListener("click", () => {
  showPage("admin");
});

cameraHomeEl.addEventListener("click", () => {
  showPage("home");
});

adminHomeEl.addEventListener("click", () => {
  showPage("home");
});

startFeedEl.addEventListener("click", () => {
  void toggleFeed();
});

directionResetEl.addEventListener("click", () => {
  recenterDirectionOrigin();
});

themeToggleEl.addEventListener("click", () => {
  const nextTheme: ThemeName = activeTheme === "dark" ? "light" : "dark";
  setTheme(nextTheme, true);
});

cameraPreviewEl.addEventListener("loadedmetadata", () => {
  updateViewportLayout();
});

window.addEventListener("resize", () => {
  updateViewportLayout();
});

window.addEventListener("orientationchange", () => {
  updateViewportLayout();
});

window.visualViewport?.addEventListener("resize", () => {
  updateViewportLayout();
});

window.setInterval(() => {
  if (currentPage !== "admin") {
    return;
  }
  void refreshAdminData();
}, 2000);

window.setInterval(() => {
  if (currentPage !== "admin") {
    return;
  }

  void refreshAdminPreviewSnapshot();
}, ADMIN_PREVIEW_POLL_MS);

window.addEventListener("beforeunload", () => {
  stopDirectionalTelemetry();
  stopStream(activeStream);
  void webrtc.disconnect();
});

showPage("home");
setTheme(activeTheme, false);
renderStartButton();
renderConnectionBadge();
renderLatestDashboard();
renderRelativeDirectionWidget();
updateViewportLayout();

function showPage(page: PageName): void {
  currentPage = page;
  homePageEl.classList.toggle("hidden", page !== "home");
  cameraPageEl.classList.toggle("hidden", page !== "camera");
  adminPageEl.classList.toggle("hidden", page !== "admin");

  if (page === "camera") {
    updateViewportLayout();
  }

  if (page === "admin") {
    void refreshAdminData();
  }
}

async function toggleFeed(): Promise<void> {
  if (activeStream) {
    await stopFeed(true);
    return;
  }

  await startFeed();
}

async function startFeed(): Promise<void> {
  startFeedEl.disabled = true;
  resetRelativeDirectionState();
  renderRelativeDirectionWidget();

  try {
    await resolveDirectionalPermissions();
    activeStream = await startCameraStream(sendTargetFps);
    attachStreamToCameraPreview(activeStream);
    await applyViewportCameraConstraints();
    logger.log("Camera stream started");

    await webrtc.connect(activeStream);
    activeSessionId = webrtc.getSessionId();
    startDirectionalTelemetry();

    if (activeSessionId) {
      logger.log(`Active session ${activeSessionId}`);
    }

    await applySendFramerate(sendTargetFps, false);
    await refreshAdminData();
  } catch (error) {
    logger.log(`Start feed failed: ${String(error)}`);
    await stopFeed(false);
  } finally {
    startFeedEl.disabled = false;
    renderStartButton();
    renderLatestDashboard();
  }
}

async function stopFeed(shouldLog: boolean): Promise<void> {
  const stream = activeStream;
  activeStream = null;

  stopDirectionalTelemetry();
  resetRelativeDirectionState();
  await webrtc.disconnect();
  if (stream) {
    stopStream(stream);
  }

  activeSessionId = null;
  attachStreamToCameraPreview(null);
  latestSessions = [];
  latestSelectedSession = null;
  syncAdminPreviewSession(null);

  if (viewportConstraintTimer !== null) {
    window.clearTimeout(viewportConstraintTimer);
    viewportConstraintTimer = null;
  }

  if (shouldLog) {
    logger.log("Feed stopped");
  }

  renderStartButton();
  renderRelativeDirectionWidget();
  renderLatestDashboard();
}

function attachStreamToCameraPreview(stream: MediaStream | null): void {
  cameraPreviewEl.srcObject = stream;

  if (stream) {
    void cameraPreviewEl.play().catch(() => undefined);
  }

  if (!stream) {
    cameraPageEl.dataset.videoOrientation = "unknown";
  }

  updateViewportLayout();
}

function setTheme(theme: ThemeName, persist: boolean): void {
  activeTheme = theme;
  document.documentElement.dataset.theme = theme;
  themeToggleEl.textContent = theme === "dark" ? "Light mode" : "Dark mode";
  themeToggleEl.setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
  );
  themeToggleEl.setAttribute("aria-pressed", theme === "light" ? "true" : "false");

  if (persist) {
    saveStoredTheme(theme);
  }
}

function updateViewportLayout(): void {
  const viewportWidth = window.visualViewport?.width ?? window.innerWidth;
  const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
  document.documentElement.style.setProperty("--app-vh", `${Math.round(viewportHeight)}px`);

  cameraPageEl.dataset.orientation =
    viewportHeight >= viewportWidth ? "portrait" : "landscape";

  if (cameraPreviewEl.videoWidth <= 0 || cameraPreviewEl.videoHeight <= 0) {
    cameraPageEl.dataset.videoOrientation = "unknown";
    return;
  }

  cameraPageEl.dataset.videoOrientation =
    cameraPreviewEl.videoHeight >= cameraPreviewEl.videoWidth
      ? "portrait"
      : "landscape";

  if (!activeStream) {
    return;
  }

  if (viewportConstraintTimer !== null) {
    window.clearTimeout(viewportConstraintTimer);
  }

  viewportConstraintTimer = window.setTimeout(() => {
    viewportConstraintTimer = null;
    void applyViewportCameraConstraints();
  }, 120);
}

async function applyViewportCameraConstraints(): Promise<void> {
  const track = activeStream?.getVideoTracks()[0] ?? null;
  if (!track) {
    return;
  }

  const viewportWidth = window.visualViewport?.width ?? window.innerWidth;
  const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
  const viewportIsPortrait = viewportHeight >= viewportWidth;

  const idealWidth = viewportIsPortrait ? 720 : 1280;
  const idealHeight = viewportIsPortrait ? 1280 : 720;
  const idealAspectRatio = viewportIsPortrait ? 9 / 16 : 16 / 9;

  try {
    await track.applyConstraints({
      width: { ideal: idealWidth },
      height: { ideal: idealHeight },
      aspectRatio: { ideal: idealAspectRatio },
      frameRate: { ideal: sendTargetFps, max: sendTargetFps }
    });
  } catch {
    return;
  }
}

function renderStartButton(): void {
  startFeedEl.textContent = activeStream ? "Stop Feed" : "Start Feed";
}

function renderConnectionBadge(): void {
  const state = peerState.toLowerCase();
  let label = "Disconnected";
  let tone: StatusTone = "neutral";

  if (state === "connected") {
    label = "Connected";
    tone = "up";
  } else if (state === "new" || state === "connecting") {
    label = "Connecting";
    tone = "warn";
  } else if (state === "failed") {
    label = "Failed";
    tone = "down";
  }

  connectionBadgeEl.textContent = label;
  connectionBadgeEl.classList.remove("is-up", "is-warn", "is-down", "is-neutral");
  connectionBadgeEl.classList.add(`is-${tone}`);
}

function renderLatestDashboard(): void {
  renderSystemDashboard(latestHealthStatus, latestSessions, latestSelectedSession);
}

function renderSystemDashboard(
  healthStatus: string,
  sessions: DebugSession[],
  selectedSession: DebugSession | null
): void {
  const apiLabel = healthStatus === "ok" ? "UP" : healthStatus === "unreachable" ? "DOWN" : "CHECKING";
  setBadgeCell(statusApiEl, apiLabel, toneForHealth(healthStatus));

  const peerDisplay = selectedSession ? selectedSession.connection_state : peerState;
  const iceDisplay = selectedSession ? selectedSession.ice_state : "n/a";
  setBadgeCell(statusPeerEl, humanizeState(peerDisplay), toneForPeerState(peerDisplay));
  setBadgeCell(statusIceEl, humanizeState(iceDisplay), toneForIceState(iceDisplay));

  setTextCell(statusSessionIdEl, selectedSession ? selectedSession.session_id : "none", true);
  setTextCell(statusSessionCountEl, String(sessions.length), true);
  setTextCell(statusSendFpsEl, `${sendTargetFps} FPS`);

  const analysisFps = selectedSession ? selectedSession.analysis_target_fps : 0;
  setTextCell(statusAnalysisFpsEl, `${formatFps(analysisFps)} FPS`);

  const incomingFps = selectedSession ? selectedSession.incoming_fps : 0;
  const processedFps = selectedSession ? selectedSession.processed_fps : 0;
  setTextCell(statusIncomingFpsEl, `${formatFps(incomingFps)} FPS`, true);
  setTextCell(statusProcessedFpsEl, `${formatFps(processedFps)} FPS`, true);
  setTextCell(
    statusDroppedFramesEl,
    String(selectedSession ? selectedSession.dropped_frames : 0),
    true
  );
  setTextCell(
    statusTotalFramesEl,
    String(selectedSession ? selectedSession.total_frames : 0),
    true
  );

  const directionalContext = selectedSession?.latest_processed_directional ?? null;
  const directionalSampleCount = selectedSession?.directional_samples_received ?? 0;
  if (!selectedSession) {
    setBadgeCell(statusDirectionalLinkEl, "N/A", "neutral");
  } else {
    setBadgeCell(
      statusDirectionalLinkEl,
      formatDirectionalLinkLabel(directionalContext, directionalSampleCount),
      toneForDirectionalLink(directionalContext, directionalSampleCount)
    );
  }

  setTextCell(statusDirectionalAgeEl, selectedSession ? formatDirectionalAge(directionalContext) : "n/a", true);
  setTextCell(
    statusDirectionalSamplesEl,
    String(selectedSession ? selectedSession.directional_samples_received : 0),
    true
  );
  setTextCell(
    statusDirectionalErrorsEl,
    String(selectedSession ? selectedSession.directional_parse_errors : 0),
    true
  );
  setTextCell(
    statusDirectionalIgnoredEl,
    String(selectedSession ? selectedSession.directional_messages_ignored : 0),
    true
  );
}

async function applySendFramerate(fps: number, shouldLog: boolean): Promise<void> {
  if (!Number.isFinite(fps) || fps < 1) {
    return;
  }

  const track = activeStream?.getVideoTracks()[0] ?? null;
  let trackUpdated = false;
  if (track) {
    try {
      await track.applyConstraints({ frameRate: { ideal: fps, max: fps } });
      trackUpdated = true;
    } catch (error) {
      logger.log(`Camera FPS update failed: ${String(error)}`);
    }
  }

  const senderUpdated = await webrtc.setOutgoingFramerate(fps);
  if (shouldLog) {
    if (senderUpdated || trackUpdated) {
      logger.log(`Send FPS cap set to ${fps}`);
    } else {
      logger.log(`Send FPS preference saved: ${fps}`);
    }
  }
}

async function refreshAdminData(): Promise<void> {
  const [healthStatus, sessions] = await Promise.all([
    fetchHealthStatus(),
    fetchDebugSessions()
  ]);

  const selectedSession = pickVisibleSession(sessions);
  latestHealthStatus = healthStatus;
  latestSessions = sessions;
  latestSelectedSession = selectedSession;
  syncAdminPreviewSession(selectedSession);

  renderLatestDashboard();
}

function syncAdminPreviewSession(selectedSession: DebugSession | null): void {
  adminPreviewSessionId = selectedSession?.session_id ?? null;
  adminPreviewHasSnapshot = Boolean(selectedSession?.has_snapshot);

  if (!adminPreviewSessionId) {
    clearAdminLivePreview("No active session.");
    return;
  }

  if (!adminPreviewHasSnapshot) {
    clearAdminLivePreview("Session active. Waiting for first frame...");
    return;
  }

  if (adminLiveStatusEl.hidden === false) {
    showAdminLiveStatus("Loading live feed...");
  }

  void refreshAdminPreviewSnapshot();
}

async function refreshAdminPreviewSnapshot(): Promise<void> {
  if (!adminPreviewSessionId || !adminPreviewHasSnapshot) {
    return;
  }

  const snapshotUrl = `${apiBaseUrl}/debug/sessions/${encodeURIComponent(
    adminPreviewSessionId
  )}/latest.jpg?ts=${Date.now()}`;
  adminLivePreviewEl.src = snapshotUrl;
}

function clearAdminLivePreview(message: string): void {
  adminLivePreviewEl.hidden = true;
  adminLivePreviewEl.removeAttribute("src");
  showAdminLiveStatus(message);
}

function showAdminLiveStatus(message: string): void {
  adminLiveStatusEl.textContent = message;
  adminLiveStatusEl.hidden = false;
}

function resetRelativeDirectionState(): void {
  headingOriginAlphaDeg = null;
  headingCurrentAlphaDeg = null;
  headingSmoothedAlphaDeg = null;
  pitchOriginBetaDeg = null;
  pitchCurrentBetaDeg = null;
  pitchSmoothedBetaDeg = null;
  displayedRelativeHeadingDeg = null;
  displayedRelativePitchDeg = null;
}

function recenterDirectionOrigin(): void {
  const recenterHeading = headingSmoothedAlphaDeg ?? headingCurrentAlphaDeg;
  const recenterPitch = pitchSmoothedBetaDeg ?? pitchCurrentBetaDeg;
  const canRecenterHeading = recenterHeading !== null;
  const canRecenterPitch = recenterPitch !== null;
  if (!canRecenterHeading && !canRecenterPitch) {
    renderRelativeDirectionWidget();
    return;
  }

  if (recenterHeading !== null) {
    headingOriginAlphaDeg = recenterHeading;
  }
  if (recenterPitch !== null) {
    pitchOriginBetaDeg = recenterPitch;
  }
  displayedRelativeHeadingDeg = 0;
  displayedRelativePitchDeg = 0;
  renderRelativeDirectionWidget();
  logger.log("Direction origin recentered");
}

function renderRelativeDirectionWidget(): void {
  const headingSample = headingSmoothedAlphaDeg ?? headingCurrentAlphaDeg;
  const pitchSample = pitchSmoothedBetaDeg ?? pitchCurrentBetaDeg;
  const hasDirectionalSample = headingSample !== null || pitchSample !== null;
  directionResetEl.disabled = !activeStream || !hasDirectionalSample;

  if (!activeStream) {
    displayedRelativeHeadingDeg = null;
    displayedRelativePitchDeg = null;
    updateRelativeHeadingDisplay(
      null,
      "--",
      "Start feed to calibrate origin."
    );
    updateRelativePitchDisplay(
      null,
      "Pitch --",
      "^v At origin",
      "Keep device level to calibrate tilt."
    );
    return;
  }

  const unavailableReason = getDirectionalUnavailableReason();

  if (headingSample === null || headingOriginAlphaDeg === null) {
    displayedRelativeHeadingDeg = null;
    updateRelativeHeadingDisplay(
      null,
      "--",
      unavailableReason ?? "Calibrating heading..."
    );
  } else {
    const relativeHeadingRaw = computeRelativeHeadingDegrees(
      headingSample,
      headingOriginAlphaDeg
    );
    const relativeHeading = applyHeadingDeadband(
      relativeHeadingRaw,
      displayedRelativeHeadingDeg,
      HEADING_JITTER_DEADBAND_DEG
    );
    displayedRelativeHeadingDeg = relativeHeading;
    updateRelativeHeadingDisplay(
      relativeHeading,
      formatSignedDegrees(relativeHeading),
      describeRelativeHeading(relativeHeading)
    );
  }

  if (pitchSample === null || pitchOriginBetaDeg === null) {
    displayedRelativePitchDeg = null;
    updateRelativePitchDisplay(
      null,
      "Pitch --",
      "^v Calibrating",
      unavailableReason ?? "Calibrating tilt..."
    );
    return;
  }

  const relativePitchRaw = computeRelativePitchDegrees(
    pitchSample,
    pitchOriginBetaDeg
  );
  const relativePitch = applyLinearDeadband(
    relativePitchRaw,
    displayedRelativePitchDeg,
    PITCH_JITTER_DEADBAND_DEG
  );
  displayedRelativePitchDeg = relativePitch;
  updateRelativePitchDisplay(
    relativePitch,
    `Pitch ${formatSignedDegrees(relativePitch)}`,
    describeRelativePitchArrow(relativePitch),
    describeRelativePitch(relativePitch)
  );
}

function updateRelativeHeadingDisplay(
  relativeHeadingDeg: number | null,
  valueLabel: string,
  detailLabel: string
): void {
  const rotation = relativeHeadingDeg ?? 0;
  directionNeedleEl.style.transform = `translate(-50%, -100%) rotate(${rotation.toFixed(1)}deg)`;
  directionRelativeValueEl.textContent = valueLabel;
  directionRelativeTextEl.textContent = detailLabel;
}

function updateRelativePitchDisplay(
  relativePitchDeg: number | null,
  valueLabel: string,
  arrowLabel: string,
  detailLabel: string
): void {
  directionPitchValueEl.textContent = valueLabel;
  directionPitchArrowEl.textContent = arrowLabel;
  directionPitchTextEl.textContent = detailLabel;

  const hasPitch = relativePitchDeg !== null;
  directionPitchValueEl.classList.toggle("is-unavailable", !hasPitch);
}

function getDirectionalUnavailableReason(): string | null {
  const orientationUnsupported = orientationPermissionState === "unsupported";
  const motionUnsupported = motionPermissionState === "unsupported";
  if (orientationUnsupported && motionUnsupported) {
    return "Direction unavailable on this browser.";
  }

  const orientationDenied = orientationPermissionState === "denied";
  const motionDenied = motionPermissionState === "denied";
  if (orientationDenied && motionDenied) {
    return "Direction blocked: sensor permission denied.";
  }

  return null;
}

function normalizeHeadingDegrees(value: number): number {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

function clampSmoothingAlpha(alpha: number): number {
  if (!Number.isFinite(alpha)) {
    return 1;
  }
  if (alpha <= 0) {
    return 0;
  }
  if (alpha >= 1) {
    return 1;
  }
  return alpha;
}

function toSignedHeadingDelta(deltaDeg: number): number {
  const normalizedDelta = normalizeHeadingDegrees(deltaDeg);
  return normalizedDelta > 180 ? normalizedDelta - 360 : normalizedDelta;
}

function smoothCircularDegrees(
  previousDeg: number | null,
  nextDeg: number,
  alpha: number
): number {
  if (previousDeg === null) {
    return nextDeg;
  }

  const blend = clampSmoothingAlpha(alpha);
  const signedDelta = toSignedHeadingDelta(nextDeg - previousDeg);
  return normalizeHeadingDegrees(previousDeg + signedDelta * blend);
}

function smoothLinearDegrees(
  previousDeg: number | null,
  nextDeg: number,
  alpha: number
): number {
  if (previousDeg === null) {
    return nextDeg;
  }

  const blend = clampSmoothingAlpha(alpha);
  return previousDeg + (nextDeg - previousDeg) * blend;
}

function applyHeadingDeadband(
  nextDeg: number,
  previousDeg: number | null,
  deadbandDeg: number
): number {
  if (previousDeg === null) {
    return nextDeg;
  }

  const signedDelta = toSignedHeadingDelta(nextDeg - previousDeg);
  return Math.abs(signedDelta) < deadbandDeg ? previousDeg : nextDeg;
}

function applyLinearDeadband(
  nextDeg: number,
  previousDeg: number | null,
  deadbandDeg: number
): number {
  if (previousDeg === null) {
    return nextDeg;
  }

  return Math.abs(nextDeg - previousDeg) < deadbandDeg ? previousDeg : nextDeg;
}

function computeRelativeHeadingDegrees(currentDeg: number, originDeg: number): number {
  return toSignedHeadingDelta(currentDeg - originDeg);
}

function computeRelativePitchDegrees(currentDeg: number, originDeg: number): number {
  let delta = currentDeg - originDeg;
  if (delta > 180) {
    delta -= 360;
  }
  if (delta < -180) {
    delta += 360;
  }
  return delta;
}

function formatSignedDegrees(valueDeg: number): string {
  const rounded = Math.round(valueDeg);
  return `${rounded > 0 ? "+" : ""}${rounded} deg`;
}

function describeRelativeHeading(relativeHeadingDeg: number): string {
  const rounded = Math.round(relativeHeadingDeg);
  if (Math.abs(rounded) <= RELATIVE_HEADING_CENTER_BAND_DEG) {
    return "Facing original direction.";
  }

  const directionWord = rounded > 0 ? "right" : "left";
  return `${Math.abs(rounded)} deg ${directionWord} of origin.`;
}

function describeRelativePitch(relativePitchDeg: number): string {
  const rounded = Math.round(relativePitchDeg);
  if (Math.abs(rounded) <= RELATIVE_PITCH_CENTER_BAND_DEG) {
    return "Level with original angle.";
  }

  const directionWord = rounded > 0 ? "up" : "down";
  return `${Math.abs(rounded)} deg ${directionWord} of origin.`;
}

function describeRelativePitchArrow(relativePitchDeg: number): string {
  const rounded = Math.round(relativePitchDeg);
  if (Math.abs(rounded) <= RELATIVE_PITCH_CENTER_BAND_DEG) {
    return "^v At origin";
  }

  return rounded > 0 ? "^ Up from origin" : "v Down from origin";
}

async function fetchHealthStatus(): Promise<string> {
  try {
    const response = await fetch(`${apiBaseUrl}/health`);
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }

    const payload = (await response.json()) as { status?: string };
    clearError("health");
    return payload.status ?? "unknown";
  } catch (error) {
    logErrorOnce("health", `Health check failed: ${String(error)}`);
    return "unreachable";
  }
}

async function fetchDebugSessions(): Promise<DebugSession[]> {
  try {
    const response = await fetch(`${apiBaseUrl}/debug/sessions`);
    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }

    const payload = (await response.json()) as { sessions?: unknown[] };
    const normalized: DebugSession[] = [];

    for (const rawSession of payload.sessions ?? []) {
      if (
        typeof rawSession !== "object" ||
        rawSession === null ||
        typeof (rawSession as RawDebugSession).session_id !== "string"
      ) {
        continue;
      }

      normalized.push(normalizeDebugSession(rawSession as RawDebugSession));
    }

    clearError("sessions");
    return normalized;
  } catch (error) {
    logErrorOnce("sessions", `Session debug fetch failed: ${String(error)}`);
    return [];
  }
}

function pickVisibleSession(
  sessions: DebugSession[],
): DebugSession | null {
  if (activeSessionId) {
    const active = sessions.find((session) => session.session_id === activeSessionId);
    if (active) {
      return active;
    }
  }

  return sessions.length > 0 ? sessions[0] : null;
}

function logErrorOnce(key: string, message: string): void {
  if (errorCache.get(key) === message) {
    return;
  }

  errorCache.set(key, message);
  logger.log(message);
}

function clearError(key: string): void {
  errorCache.delete(key);
}

async function resolveDirectionalPermissions(): Promise<void> {
  if (directionalPermissionsResolved) {
    return;
  }

  const windowWithSensorConstructors = window as Window & {
    DeviceMotionEvent?: unknown;
    DeviceOrientationEvent?: unknown;
  };

  const motionConstructor = windowWithSensorConstructors.DeviceMotionEvent;
  const orientationConstructor = windowWithSensorConstructors.DeviceOrientationEvent;

  motionPermissionState = await requestSensorPermission(motionConstructor);
  orientationPermissionState = await resolveOrientationPermission(
    orientationConstructor,
    motionPermissionState
  );
  directionalPermissionsResolved = true;

  if (motionPermissionState === "denied" && orientationPermissionState === "denied") {
    logger.log("Directional telemetry unavailable: sensor permission denied.");
  }
}

async function resolveOrientationPermission(
  orientationConstructor: unknown,
  motionPermission: SensorPermissionState
): Promise<SensorPermissionState> {
  if (!orientationConstructor) {
    return "unsupported";
  }

  if (hasPermissionRequestMethod(orientationConstructor)) {
    if (motionPermission === "granted" || motionPermission === "denied") {
      return motionPermission;
    }
  }

  return requestSensorPermission(orientationConstructor);
}

function hasPermissionRequestMethod(
  eventConstructor: unknown
): eventConstructor is { requestPermission: () => Promise<string> } {
  return (
    (typeof eventConstructor === "function" || typeof eventConstructor === "object") &&
    eventConstructor !== null &&
    typeof (eventConstructor as { requestPermission?: unknown }).requestPermission ===
      "function"
  );
}

async function requestSensorPermission(eventConstructor: unknown): Promise<SensorPermissionState> {
  if (!eventConstructor) {
    return "unsupported";
  }

  if (!hasPermissionRequestMethod(eventConstructor)) {
    return "granted";
  }

  try {
    const result = await eventConstructor.requestPermission();
    return result === "granted" ? "granted" : "denied";
  } catch {
    return "denied";
  }
}

function startDirectionalTelemetry(): void {
  if (directionalTelemetryActive) {
    return;
  }

  const canListenMotion = motionPermissionState === "granted";
  const canListenOrientation = orientationPermissionState === "granted";
  if (!canListenMotion && !canListenOrientation) {
    if (
      motionPermissionState === "unsupported" &&
      orientationPermissionState === "unsupported"
    ) {
      logger.log("Directional telemetry unavailable: no sensor support in this browser.");
    }
    renderRelativeDirectionWidget();
    return;
  }

  directionalTelemetryActive = true;
  latestRotationRateDps = null;
  latestOrientationDegrees = null;
  lastDirectionalTelemetrySentAtMs = 0;

  if (canListenMotion) {
    window.addEventListener("devicemotion", handleDirectionalDeviceMotion, {
      passive: true
    });
  }
  if (canListenOrientation) {
    window.addEventListener("deviceorientation", handleDirectionalDeviceOrientation, {
      passive: true
    });
  }

  const sources: string[] = [];
  if (canListenMotion) {
    sources.push("gyro");
  }
  if (canListenOrientation) {
    sources.push("orientation");
  }
  logger.log(`Directional telemetry active (${sources.join(", ")})`);
  renderRelativeDirectionWidget();
}

function stopDirectionalTelemetry(): void {
  if (!directionalTelemetryActive) {
    return;
  }

  directionalTelemetryActive = false;
  window.removeEventListener("devicemotion", handleDirectionalDeviceMotion);
  window.removeEventListener("deviceorientation", handleDirectionalDeviceOrientation);
  latestRotationRateDps = null;
  latestOrientationDegrees = null;
  lastDirectionalTelemetrySentAtMs = 0;
  renderRelativeDirectionWidget();
}

function handleDirectionalDeviceMotion(event: DeviceMotionEvent): void {
  if (!directionalTelemetryActive) {
    return;
  }

  const rotationRate = event.rotationRate;
  if (!rotationRate) {
    return;
  }

  latestRotationRateDps = {
    alpha: toFiniteNumberOrNull(rotationRate.alpha),
    beta: toFiniteNumberOrNull(rotationRate.beta),
    gamma: toFiniteNumberOrNull(rotationRate.gamma)
  };
  maybeSendDirectionalTelemetry(Date.now());
}

function handleDirectionalDeviceOrientation(event: DeviceOrientationEvent): void {
  if (!directionalTelemetryActive) {
    return;
  }

  const normalizedHeading = toFiniteNumberOrNull(event.alpha);
  const normalizedPitch = toFiniteNumberOrNull(event.beta);
  if (normalizedHeading !== null) {
    headingCurrentAlphaDeg = normalizeHeadingDegrees(normalizedHeading);
    headingSmoothedAlphaDeg = smoothCircularDegrees(
      headingSmoothedAlphaDeg,
      headingCurrentAlphaDeg,
      HEADING_SMOOTHING_ALPHA
    );
    if (headingOriginAlphaDeg === null) {
      headingOriginAlphaDeg = headingSmoothedAlphaDeg;
    }
  }
  if (normalizedPitch !== null) {
    pitchCurrentBetaDeg = normalizedPitch;
    pitchSmoothedBetaDeg = smoothLinearDegrees(
      pitchSmoothedBetaDeg,
      pitchCurrentBetaDeg,
      PITCH_SMOOTHING_ALPHA
    );
    if (pitchOriginBetaDeg === null) {
      pitchOriginBetaDeg = pitchSmoothedBetaDeg;
    }
  }

  latestOrientationDegrees = {
    alpha: normalizedHeading,
    beta: normalizedPitch,
    gamma: toFiniteNumberOrNull(event.gamma),
    absolute: typeof event.absolute === "boolean" ? event.absolute : null
  };
  renderRelativeDirectionWidget();
  maybeSendDirectionalTelemetry(Date.now());
}

function maybeSendDirectionalTelemetry(timestampMs: number): void {
  if (!directionalTelemetryActive) {
    return;
  }

  if (timestampMs - lastDirectionalTelemetrySentAtMs < SENSOR_TELEMETRY_INTERVAL_MS) {
    return;
  }

  const payload = buildDirectionalTelemetryPayload(timestampMs);
  if (!payload) {
    return;
  }

  if (webrtc.sendJson(payload)) {
    lastDirectionalTelemetrySentAtMs = timestampMs;
  }
}

function buildDirectionalTelemetryPayload(
  timestampMs: number
): DirectionalTelemetryMessage | null {
  if (!latestRotationRateDps && !latestOrientationDegrees) {
    return null;
  }

  return {
    type: "client_sensor",
    sensor: "gyro",
    timestamp_ms: timestampMs,
    rotation_rate_dps: latestRotationRateDps ? { ...latestRotationRateDps } : null,
    orientation_deg: latestOrientationDegrees ? { ...latestOrientationDegrees } : null
  };
}

function toFiniteNumberOrNull(value: number | null): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }

  return value;
}

function handleInferencePayload(rawPayload: string): void {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawPayload);
  } catch {
    logger.log(`Data message: ${rawPayload}`);
    return;
  }

  if (!isInferenceMessage(parsed)) {
    logger.log(`Data message: ${rawPayload}`);
    return;
  }

  const message = parsed as InferenceMessage;
  if (message.guidance_text) {
    logger.log(`Guidance: ${message.guidance_text}`);
    return;
  }

  logger.log(`Inference tick: ${message.timestamp}`);
}
