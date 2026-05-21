import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator


_DATE_TIME_PATTERN = re.compile(
    r"(?P<prefix>.*?)(?P<date>20\d{6})\s*[_\-\u2013\u2014]\s*(?P<time>\d{6})",
    re.IGNORECASE,
)
_P_CAMERA_PATTERN = re.compile(
    r"(?P<camera>[A-Za-z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"[_\-\u2013\u2014](?P<start>\d{6})(?:[_\-\u2013\u2014](?P<end>\d{6}))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoSegment:
    """Metadata for one physical video file in a camera timeline."""

    path: str
    rel_path: str
    camera_id: str
    start_datetime: datetime | None
    end_datetime: datetime | None = None
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class StreamFrame:
    """Frame yielded from a stitched camera stream with both local and stream timing."""

    frame: object
    segment: VideoSegment
    local_frame_idx: int
    stream_frame_idx: int
    source_time_sec: float
    absolute_datetime: datetime | None


def _clean_camera_id(value: str | None, fallback: str = "camera") -> str:
    text = (value or "").strip(" _-\u2013\u2014.")
    if not text:
        return fallback
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def parse_video_filename(path: str, rel_path: str | None = None) -> VideoSegment:
    """Infer camera id and timestamps from supported CCTV filename patterns."""
    name = os.path.splitext(os.path.basename(path or ""))[0]
    rel = rel_path or os.path.basename(path or "")

    match = _P_CAMERA_PATTERN.search(name)
    if match:
        year = 2000 + int(match.group("yy"))
        start = datetime.strptime(
            f"{year:04d}{match.group('mm')}{match.group('dd')}{match.group('start')}",
            "%Y%m%d%H%M%S",
        )
        end = None
        if match.group("end"):
            end = datetime.strptime(
                f"{year:04d}{match.group('mm')}{match.group('dd')}{match.group('end')}",
                "%Y%m%d%H%M%S",
            )
            if end < start:
                end = end + timedelta(days=1)
        return VideoSegment(
            path=path,
            rel_path=rel,
            camera_id=_clean_camera_id(match.group("camera")),
            start_datetime=start,
            end_datetime=end,
        )

    match = _DATE_TIME_PATTERN.search(name)
    if match:
        start = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")
        camera_id = _clean_camera_id(match.group("prefix"), fallback="camera")
        return VideoSegment(
            path=path,
            rel_path=rel,
            camera_id=camera_id,
            start_datetime=start,
        )

    parent = os.path.basename(os.path.dirname(path or "")) or "camera"
    return VideoSegment(
        path=path,
        rel_path=rel,
        camera_id=_clean_camera_id(parent),
        start_datetime=None,
    )


def load_video_segment_metadata(segment: VideoSegment) -> VideoSegment:
    """Open the video once to fill FPS, frame count, dimensions, and inferred end time."""
    import cv2

    cap = cv2.VideoCapture(segment.path)
    if not cap.isOpened():
        return segment
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    end = segment.end_datetime
    if end is None and segment.start_datetime is not None and fps > 0 and frame_count > 0:
        end = segment.start_datetime + timedelta(seconds=frame_count / fps)

    return VideoSegment(
        path=segment.path,
        rel_path=segment.rel_path,
        camera_id=segment.camera_id,
        start_datetime=segment.start_datetime,
        end_datetime=end,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )


def build_video_segments(
    video_paths: Iterable[str],
    input_dir: str | None = None,
    load_metadata: bool = True,
) -> list[VideoSegment]:
    """Build and sort video segments so each camera can be processed chronologically."""
    segments = []
    for path in video_paths:
        rel_path = os.path.relpath(path, input_dir) if input_dir else os.path.basename(path)
        segment = parse_video_filename(path, rel_path=rel_path)
        if load_metadata:
            segment = load_video_segment_metadata(segment)
        segments.append(segment)
    return sorted(
        segments,
        key=lambda item: (
            item.camera_id,
            item.start_datetime or datetime.min,
            item.rel_path,
        ),
    )


class VideoStreamServer:
    """Groups video files by camera and yields frames as continuous camera streams."""

    def __init__(self, segments: Iterable[VideoSegment]):
        self.segments = list(segments)

    @classmethod
    def from_paths(cls, video_paths: Iterable[str], input_dir: str | None = None, load_metadata: bool = True):
        return cls(build_video_segments(video_paths, input_dir=input_dir, load_metadata=load_metadata))

    @property
    def total_frames(self) -> int:
        return sum(max(0, int(segment.frame_count or 0)) for segment in self.segments)

    @property
    def camera_ids(self) -> list[str]:
        return sorted({segment.camera_id for segment in self.segments})

    def iter_camera_segments(self) -> Iterator[tuple[str, list[VideoSegment]]]:
        current_camera = None
        current_segments = []
        for segment in self.segments:
            if current_camera is None:
                current_camera = segment.camera_id
            if segment.camera_id != current_camera:
                yield current_camera, current_segments
                current_camera = segment.camera_id
                current_segments = []
            current_segments.append(segment)
        if current_camera is not None:
            yield current_camera, current_segments

    def frames_for_camera(self, camera_id: str, segments: Iterable[VideoSegment]) -> Iterator[StreamFrame]:
        import cv2

        # stream_frame_idx resets per camera; local_frame_idx resets for each segment.
        stream_frame_idx = 0
        for segment in segments:
            cap = cv2.VideoCapture(segment.path)
            if not cap.isOpened():
                continue
            fps = segment.fps or float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
            local_frame_idx = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    local_frame_idx += 1
                    stream_frame_idx += 1
                    source_time_sec = local_frame_idx / fps
                    absolute_datetime = None
                    if segment.start_datetime is not None:
                        absolute_datetime = segment.start_datetime + timedelta(seconds=source_time_sec)
                    yield StreamFrame(
                        frame=frame,
                        segment=segment,
                        local_frame_idx=local_frame_idx,
                        stream_frame_idx=stream_frame_idx,
                        source_time_sec=source_time_sec,
                        absolute_datetime=absolute_datetime,
                    )
            finally:
                cap.release()
