"""Root-level compatibility launcher for the legacy Tk backend."""

from cctv_roi_ai_event_extractor.event_processing import *  # noqa: F401,F403
from cctv_roi_ai_event_extractor.event_processing import main
from cctv_roi_ai_event_extractor.legacy_app import App, ParamsDialog, PastePathsDialog  # noqa: F401


if __name__ == "__main__":
    main()
