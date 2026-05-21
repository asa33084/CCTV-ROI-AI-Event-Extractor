"""Root-level compatibility launcher for the legacy Tk backend."""

from cctv_roi_ai_event_extractor.event_processing import *  # noqa: F401,F403
from cctv_roi_ai_event_extractor.event_processing import main


if __name__ == "__main__":
    main()
