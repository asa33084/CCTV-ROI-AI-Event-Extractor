import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Iterator

CONTINUOUS_STREAM_GAP_SEC = 2.0


_DATE_TIME_PATTERN = re.compile(
    r"(?P<prefix>.*?)(?P<date>20\d{6})\s*[_\-\u2013\u2014]\s*(?P<time>\d{6})",
    re.IGNORECASE,
)
_P_CAMERA_PATTERN = re.compile(
    r"(?P<camera>[A-Za-z]+)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"[_\-\u2013\u2014](?P<start>\d{6})(?:[_\-\u2013\u2014](?P<end>\d{6}))?",
    re.IGNORECASE,
)
_ISO_Z_CAMERA_PATTERN = re.compile(
    r"(?P<date>20\d{2}-\d{2}-\d{2})T(?P<time>\d{2}-\d{2}-\d{2})Z_"
    r"(?P<camera>[^_]+)(?:_.*)?$",
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
class VideoStream:
    """A chronological, gap-free stream for one camera."""

    camera_id: str
    stream_id: str
    segments: tuple[VideoSegment, ...]
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None


@dataclass(frozen=True)
class StreamFrame:
    """Frame yielded from a stitched camera stream with both local and stream timing."""

    frame: object
    stream: VideoStream
    segment: VideoSegment
    local_frame_idx: int
    stream_frame_idx: int
    source_time_sec: float
    stream_time_sec: float
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

    match = _ISO_Z_CAMERA_PATTERN.search(name)
    if match:
        start = datetime.strptime(match.group("date") + match.group("time"), "%Y-%m-%d%H-%M-%S")
        return VideoSegment(
            path=path,
            rel_path=rel,
            camera_id=_clean_camera_id(match.group("camera")),
            start_datetime=start,
        )

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
    """Groups video files by camera and yields gap-free chronological streams."""

    def __init__(self, segments: Iterable[VideoSegment], max_gap_sec: float = CONTINUOUS_STREAM_GAP_SEC):
        self.segments = list(segments)
        self.max_gap_sec = float(max_gap_sec)

    @classmethod
    def from_paths(
        cls,
        video_paths: Iterable[str],
        input_dir: str | None = None,
        load_metadata: bool = True,
        max_gap_sec: float = CONTINUOUS_STREAM_GAP_SEC,
    ):
        return cls(
            build_video_segments(video_paths, input_dir=input_dir, load_metadata=load_metadata),
            max_gap_sec=max_gap_sec,
        )

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

    def iter_streams(self) -> Iterator[VideoStream]:
        """Yield one stream per camera timeline chunk separated by time gaps."""
        for camera_id, segments in self.iter_camera_segments():
            current_segments = []
            stream_index = 1
            previous_segment = None
            for segment in segments:
                gap_sec = _segment_gap_sec(previous_segment, segment)
                should_split = current_segments and (
                    gap_sec is None or gap_sec > self.max_gap_sec
                )
                if should_split:
                    yield _make_video_stream(camera_id, stream_index, current_segments)
                    stream_index += 1
                    current_segments = []
                current_segments.append(segment)
                previous_segment = segment
            if current_segments:
                yield _make_video_stream(camera_id, stream_index, current_segments)

    def frames_for_stream(self, stream: VideoStream) -> Iterator[StreamFrame]:
        import cv2

        # stream_frame_idx resets per continuous stream; local_frame_idx resets per file.
        stream_frame_idx = 0
        stream_elapsed_sec = 0.0
        for segment in stream.segments:
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
                    stream_time_sec = stream_elapsed_sec + source_time_sec
                    absolute_datetime = None
                    if segment.start_datetime is not None:
                        absolute_datetime = segment.start_datetime + timedelta(seconds=source_time_sec)
                    yield StreamFrame(
                        frame=frame,
                        stream=stream,
                        segment=segment,
                        local_frame_idx=local_frame_idx,
                        stream_frame_idx=stream_frame_idx,
                        source_time_sec=source_time_sec,
                        stream_time_sec=stream_time_sec,
                        absolute_datetime=absolute_datetime,
                    )
            finally:
                cap.release()
            stream_elapsed_sec += _segment_duration_sec(segment)

    def frames_for_camera(self, camera_id: str, segments: Iterable[VideoSegment]) -> Iterator[StreamFrame]:
        """Compatibility wrapper; prefer iter_streams() + frames_for_stream()."""
        stream = _make_video_stream(camera_id, 1, list(segments))
        yield from self.frames_for_stream(stream)


def _segment_gap_sec(previous_segment, current_segment):
    if (
        previous_segment is None
        or previous_segment.end_datetime is None
        or current_segment.start_datetime is None
    ):
        return None
    return (current_segment.start_datetime - previous_segment.end_datetime).total_seconds()


def _segment_duration_sec(segment: VideoSegment) -> float:
    if segment.fps > 0 and segment.frame_count > 0:
        return segment.frame_count / segment.fps
    if segment.start_datetime is not None and segment.end_datetime is not None:
        return max(0.0, (segment.end_datetime - segment.start_datetime).total_seconds())
    return 0.0


def _make_video_stream(camera_id: str, stream_index: int, segments: list[VideoSegment]) -> VideoStream:
    start = next((segment.start_datetime for segment in segments if segment.start_datetime is not None), None)
    end = next((segment.end_datetime for segment in reversed(segments) if segment.end_datetime is not None), None)
    stream_id = f"{camera_id}-S{stream_index:03d}"
    return VideoStream(
        camera_id=camera_id,
        stream_id=stream_id,
        segments=tuple(segments),
        start_datetime=start,
        end_datetime=end,
    )
