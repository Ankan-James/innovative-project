/**
 * SLMCI — Camera & Clip Recording Controller
 * Manages webcam lifecycle, MediaRecorder clip loops, and API coordination
 */

class CameraController {
  constructor(options) {
    this.videoElement = options.videoElement;
    this.placeholderElement = options.placeholderElement;
    this.statusElement = options.statusElement;
    this.startButton = options.startButton;
    this.stopButton = options.stopButton;
    this.nameModeCheckbox = options.nameModeCheckbox;
    
    this.apiClient = options.apiClient;
    this.interpreterUI = options.interpreterUI;
    this.overlay = options.overlay;

    this.clipDuration = 1500; // 1.5s as specified
    this.stream = null;
    this.recorder = null;
    this.recordingTimer = null;
    this.running = false;
    this.selectedMimeType = null;
    this.sessionId = 0;
    this.recognitionSessionId = null;
  }

  /**
   * Initializes and starts webcam streaming and recognition loop
   */
  async start() {
    if (this.running || this.startButton.disabled) return;

    this.startButton.disabled = true;
    this.setStatus("Requesting camera access...");

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Webcam access is not supported by this browser.");
      }

      if (!window.MediaRecorder) {
        throw new Error("Video recording (MediaRecorder) is unavailable.");
      }

      const supportedTypes = [
        "video/webm;codecs=vp8",
        "video/webm",
        "video/mp4"
      ];

      this.selectedMimeType = supportedTypes.find(type =>
        MediaRecorder.isTypeSupported(type)
      );

      if (!this.selectedMimeType) {
        throw new Error("No supported video recording format was found.");
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        },
        audio: false
      });

      this.videoElement.srcObject = this.stream;
      await this.videoElement.play();

      this.running = true;
      const currentSession = ++this.sessionId;

      // Unique session ID for word accumulation in FastAPI backend
      this.recognitionSessionId = crypto.randomUUID();

      this.stopButton.disabled = false;
      if (this.placeholderElement) {
        this.placeholderElement.style.opacity = "0";
        setTimeout(() => {
          if (this.running) this.placeholderElement.style.display = "none";
        }, 200);
      }

      this.interpreterUI.setWaitingForHand();
      this.setStatus("Camera active. Ready to interpret signs.");

      // Start continuous recognition loop
      this.recognitionLoop(currentSession);

    } catch (error) {
      this.stop();
      this.setStatus(`Camera error: ${error.message}`);
    }
  }

  /**
   * Records a single 1.5-second video clip
   *
   * @param {number} currentSession - Session counter to prevent stale operations
   * @returns {Promise<Blob|null>}
   */
  recordClip(currentSession) {
    return new Promise((resolve, reject) => {
      if (!this.running || currentSession !== this.sessionId || !this.stream) {
        resolve(null);
        return;
      }

      const chunks = [];

      try {
        const recorder = new MediaRecorder(this.stream, {
          mimeType: this.selectedMimeType,
          videoBitsPerSecond: 800000
        });
        this.recorder = recorder;

        recorder.ondataavailable = event => {
          if (event.data && event.data.size > 0) {
            chunks.push(event.data);
          }
        };

        recorder.onerror = () => {
          clearTimeout(this.recordingTimer);
          reject(new Error("Video recording failed."));
        };

        recorder.onstop = () => {
          clearTimeout(this.recordingTimer);
          if (this.recorder === recorder) {
            this.recorder = null;
          }
          const blob = new Blob(chunks, { type: recorder.mimeType });
          resolve(blob);
        };

        recorder.start();

        this.recordingTimer = setTimeout(() => {
          if (recorder.state === "recording") {
            recorder.stop();
          }
        }, this.clipDuration);

      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * Continuous loop recording clips and querying FastAPI backend
   *
   * @param {number} currentSession
   */
  async recognitionLoop(currentSession) {
    while (this.running && currentSession === this.sessionId) {
      try {
        this.setStatus("Recording sign gesture (1.5s)...");

        const clip = await this.recordClip(currentSession);

        if (!this.running || currentSession !== this.sessionId) break;
        if (!clip || clip.size === 0) {
          throw new Error("Recorded video clip was empty.");
        }

        this.setStatus("Analyzing sign frames with ViT...");
        this.interpreterUI.setAnalyzing();

        const isNameMode = this.nameModeCheckbox ? this.nameModeCheckbox.checked : false;
        const result = await this.apiClient.predictVideo(
          clip,
          this.recognitionSessionId,
          isNameMode
        );

        if (!this.running || currentSession !== this.sessionId) break;

        if (result) {
          // Sync overlay canvas resolution to webcam dimensions
          this.overlay.syncDimensions(this.videoElement);

          // Draw green bounding box if hand detected
          if (result.hand_detected && result.bbox) {
            this.overlay.drawBox(
              result.bbox,
              result.frame_width,
              result.frame_height
            );
          } else {
            this.overlay.clear();
          }

          // Update detector UI elements (letter, confidence, status, voting grid, text)
          this.interpreterUI.displayResult(result);

          // Update camera status summary
          if (!result.hand_detected) {
            this.setStatus("No hand detected. Position your hand inside the frame.");
          } else if (!result.label) {
            this.setStatus("Hand detected. Hold gesture steady for 5+ matching frames.");
          } else {
            this.setStatus(`Sign "${result.label}" recognized (${result.votes}/${result.frames_analyzed || 8} votes).`);
          }
        }

        // Rest interval between clips
        await new Promise(resolve => setTimeout(resolve, 300));

      } catch (error) {
        if (error.name === "AbortError") break;

        this.stop();
        this.setStatus(`Recognition error: ${error.message}`);
        break;
      }
    }
  }

  /**
   * Stops camera stream, clears recorder, aborts requests, resets canvas and UI
   */
  stop() {
    this.running = false;
    this.sessionId++;

    clearTimeout(this.recordingTimer);

    if (this.apiClient) {
      this.apiClient.abort();
    }

    if (this.recorder && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch (_) {}
    }
    this.recorder = null;

    if (this.stream) {
      this.stream.getTracks().forEach(track => {
        try {
          track.stop();
        } catch (_) {}
      });
      this.stream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    if (this.overlay) {
      this.overlay.clear();
    }

    this.startButton.disabled = false;
    this.stopButton.disabled = true;

    if (this.placeholderElement) {
      this.placeholderElement.style.display = "flex";
      this.placeholderElement.style.opacity = "1";
    }

    if (this.interpreterUI) {
      this.interpreterUI.reset();
    }

    this.setStatus("Camera stopped.");
  }

  /**
   * Resets active session ID to start fresh text formation
   */
  resetSession() {
    this.recognitionSessionId = crypto.randomUUID();
    if (this.interpreterUI) {
      this.interpreterUI.clearFormedText();
    }
  }

  setStatus(message) {
    if (this.statusElement) {
      this.statusElement.textContent = message;
    }
  }
}

if (typeof window !== "undefined") {
  window.CameraController = CameraController;
}
