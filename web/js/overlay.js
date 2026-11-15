/**
 * SLMCI — Hand Bounding Box Overlay
 * Manages canvas dimensions, coordinate scaling, and rendering
 */

class HandOverlay {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext("2d");
    this.boxColor = "#39A76E";
    this.lineWidth = 3;
  }

  /**
   * Resizes canvas to match incoming video resolution
   */
  syncDimensions(videoElement) {
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;

    if (!videoWidth || !videoHeight) return false;

    if (this.canvas.width !== videoWidth || this.canvas.height !== videoHeight) {
      this.canvas.width = videoWidth;
      this.canvas.height = videoHeight;
    }
    return true;
  }

  /**
   * Draws green bounding box on canvas.
   * Note: Canvas element uses CSS `transform: scaleX(-1)` to match mirrored video preview.
   * Standard coordinates drawn on the canvas are horizontally reflected in the viewport.
   *
   * @param {Array<number>|null} bbox - [x1, y1, x2, y2]
   * @param {number} frameWidth - Resolution width of analyzed frame
   * @param {number} frameHeight - Resolution height of analyzed frame
   */
  drawBox(bbox, frameWidth, frameHeight) {
    this.clear();

    if (!bbox || !frameWidth || !frameHeight) {
      return;
    }

    const [x1, y1, x2, y2] = bbox;
    const scaleX = this.canvas.width / frameWidth;
    const scaleY = this.canvas.height / frameHeight;

    const rectX = x1 * scaleX;
    const rectY = y1 * scaleY;
    const rectWidth = (x2 - x1) * scaleX;
    const rectHeight = (y2 - y1) * scaleY;

    this.ctx.save();
    this.ctx.strokeStyle = this.boxColor;
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.lineJoin = "round";

    // Draw bounding box
    this.ctx.strokeRect(rectX, rectY, rectWidth, rectHeight);

    // Draw subtle corner accents for a refined technical look
    const cornerSize = Math.min(rectWidth, rectHeight) * 0.15;
    this.ctx.lineWidth = this.lineWidth + 1;
    
    // Top-left corner
    this.ctx.beginPath();
    this.ctx.moveTo(rectX, rectY + cornerSize);
    this.ctx.lineTo(rectX, rectY);
    this.ctx.lineTo(rectX + cornerSize, rectY);
    this.ctx.stroke();

    // Top-right corner
    this.ctx.beginPath();
    this.ctx.moveTo(rectX + rectWidth - cornerSize, rectY);
    this.ctx.lineTo(rectX + rectWidth, rectY);
    this.ctx.lineTo(rectX + rectWidth, rectY + cornerSize);
    this.ctx.stroke();

    // Bottom-left corner
    this.ctx.beginPath();
    this.ctx.moveTo(rectX, rectY + rectHeight - cornerSize);
    this.ctx.lineTo(rectX, rectY + rectHeight);
    this.ctx.lineTo(rectX + cornerSize, rectY + rectHeight);
    this.ctx.stroke();

    // Bottom-right corner
    this.ctx.beginPath();
    this.ctx.moveTo(rectX + rectWidth - cornerSize, rectY + rectHeight);
    this.ctx.lineTo(rectX + rectWidth, rectY + rectHeight);
    this.ctx.lineTo(rectX + rectWidth, rectY + rectHeight - cornerSize);
    this.ctx.stroke();

    this.ctx.restore();
  }

  /**
   * Clears the overlay canvas
   */
  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
}

// Export for modular usage or global attachment
if (typeof window !== "undefined") {
  window.HandOverlay = HandOverlay;
}
