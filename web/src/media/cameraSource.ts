export async function startCameraStream(): Promise<MediaStream> {
  if (!("mediaDevices" in navigator) || !navigator.mediaDevices) {
    const secureHint =
      !window.isSecureContext
        ? "Camera APIs require HTTPS on most mobile browsers (or localhost)."
        : "This browser does not expose mediaDevices APIs.";
    throw new Error(`Camera API unavailable. ${secureHint}`);
  }

  if (typeof navigator.mediaDevices.getUserMedia !== "function") {
    throw new Error("getUserMedia is not available in this browser.");
  }

  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: "environment" },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 30 }
    }
  });
}

export function stopStream(stream: MediaStream | null): void {
  if (!stream) {
    return;
  }
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
