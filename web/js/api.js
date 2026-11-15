/**
 * SLMCI — FastAPI Client
 * Handles video clip transmission, multipart form payloads, and request abortion
 */

class ApiClient {
  constructor(endpoint = "/predict-video") {
    this.endpoint = endpoint;
    this.currentController = null;
  }

  /**
   * Uploads recorded video clip to FastAPI backend
   *
   * @param {Blob} videoBlob - The recorded video clip
   * @param {string} sessionId - UUID of active camera session
   * @param {boolean} nameMode - Whether autocorrect is temporarily disabled
   * @returns {Promise<Object>} Backend response JSON
   */
  async predictVideo(videoBlob, sessionId, nameMode) {
    if (!videoBlob || videoBlob.size === 0) {
      throw new Error("Recorded video clip is empty.");
    }

    // Abort previous in-flight request if any
    this.abort();

    const controller = new AbortController();
    this.currentController = controller;

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("name_mode", nameMode ? "true" : "false");

    const mime = videoBlob.type.split(";")[0];
    const extension = mime === "video/mp4" ? "mp4" : "webm";
    formData.append("video", videoBlob, `slmci_clip.${extension}`);

    try {
      const response = await fetch(this.endpoint, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      if (!response.ok) {
        let errorMessage = `HTTP error ${response.status}`;
        try {
          const errorJson = await response.json();
          if (errorJson && errorJson.detail) {
            errorMessage = errorJson.detail;
          }
        } catch (_) {
          // Fallback to HTTP status text
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      return result;

    } catch (error) {
      if (error.name === "AbortError") {
        return null;
      }
      throw error;

    } finally {
      if (this.currentController === controller) {
        this.currentController = null;
      }
    }
  }

  /**
   * Aborts in-flight request if one exists
   */
  abort() {
    if (this.currentController) {
      this.currentController.abort();
      this.currentController = null;
    }
  }
}

if (typeof window !== "undefined") {
  window.ApiClient = ApiClient;
}
