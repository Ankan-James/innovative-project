
from pathlib import Path
import tempfile
import uuid

from fastapi import UploadFile, HTTPException


# Project root directory
ROOT = Path(__file__).resolve().parents[2]

# Directory for recordings we choose to keep
RECORDINGS_DIR = ROOT / "data" / "recordings"

# Maximum video size: 20 MB
MAX_VIDEO_SIZE = 20 * 1024 * 1024

SUPPORTED_FORMATS = {
    "video/webm": ".webm",
    "video/mp4": ".mp4",
}


async def receive_video(
    video: UploadFile,
    save_permanently: bool = False
) -> Path:
    """
    Receive a video uploaded from the browser.

    By default, stores it temporarily for analysis.
    Set save_permanently=True to keep the recording.

    Returns the path to the saved video file.
    """

    content_type = (video.content_type or "").split(";")[0].lower()

    if content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415,
            detail="Only WebM and MP4 videos are supported."
        )

    extension = SUPPORTED_FORMATS[content_type]
    filename = f"slmci_{uuid.uuid4().hex}{extension}"

    if save_permanently:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECORDINGS_DIR / filename
    else:
        path = Path(tempfile.gettempdir()) / filename

    total_size = 0

    try:
        with open(path, "xb") as output:
            while chunk := await video.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds the 20 MB limit."
                    )

                output.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

        return path

    except Exception:
        path.unlink(missing_ok=True)
        raise

    finally:
        await video.close()


def delete_temporary_video(path: Path):
    """Remove a temporary video after processing."""

    temporary_directory = Path(tempfile.gettempdir()).resolve()
    path = Path(path).resolve()

    if (
        path.parent == temporary_directory
        and path.name.startswith("slmci_")
        and path.suffix in SUPPORTED_FORMATS.values()
    ):
        path.unlink(missing_ok=True)
