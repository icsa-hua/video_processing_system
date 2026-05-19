from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


# COCO-style class IDs, useful if your YOLO model outputs COCO IDs.
# COCO:
# bicycle=1, car=2, motorcycle=3, bus=5, truck=7
UA_DETRAC_TO_COCO = {
    "car": 2,
    "van": 7,
    "bus": 5,
    "truck": 7,
    "others": 2,
}


# Compact IDs, useful if you trained a custom detector with only vehicle classes.
UA_DETRAC_TO_COMPACT = {
    "car": 0,
    "van": 1,
    "bus": 2,
    "truck": 3,
    "others": 4,
}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def natural_key(path: Path):
    """
    Sort paths like:
    img1.jpg, img2.jpg, img10.jpg
    instead of:
    img1.jpg, img10.jpg, img2.jpg
    """
    text = path.stem
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def find_matching_xml(frames_dir: Path, annotations_dir: Path) -> Path:
    """
    UA-DETRAC sequence folders are often named like MVI_20011,
    and the matching annotation is usually MVI_20011.xml.
    """
    expected = annotations_dir / f"{frames_dir.name}.xml"
    if expected.exists():
        return expected

    matches = list(annotations_dir.glob(f"*{frames_dir.name}*.xml"))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find matching XML for frame folder '{frames_dir.name}' "
        f"in annotation dir '{annotations_dir}'. Expected: {expected}"
    )


def collect_frames(frames_dir: Path) -> list[Path]:
    frames = [
        p for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    frames = sorted(frames, key=natural_key)

    if not frames:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    return frames


def create_video_from_frames(
    frames: list[Path],
    output_video: Path,
    fps: float = 25.0,
    codec: str = "mp4v",
) -> tuple[int, int]:
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")

    height, width = first.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_video}")

    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)

    writer.release()
    return width, height


def parse_uadetrac_xml_to_yolo(
    xml_path: Path,
    labels_dir: Path,
    frame_count: int,
    image_width: int,
    image_height: int,
    class_mode: str = "coco",
    label_name_mode: str = "frame_index",
    frame_paths: list[Path] | None = None,
    skip_ignored_regions: bool = True,
    skip_unknown_classes: bool = True,
) -> None:
    """
    Converts UA-DETRAC XML annotations to YOLO format.

    YOLO format:
    class_id x_center_norm y_center_norm width_norm height_norm
    """

    if class_mode == "coco":
        class_map = UA_DETRAC_TO_COCO
    elif class_mode == "compact":
        class_map = UA_DETRAC_TO_COMPACT
    else:
        raise ValueError("class_mode must be either 'coco' or 'compact'")

    labels_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Create empty label files for all frames.
    # This is important because frames with no objects should still have an empty .txt.
    for idx in range(1, frame_count + 1):
        label_path = make_label_path(
            labels_dir=labels_dir,
            frame_idx=idx,
            label_name_mode=label_name_mode,
            frame_paths=frame_paths,
        )
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("", encoding="utf-8")

    for frame_elem in root.findall("frame"):
        frame_num = int(frame_elem.attrib["num"])

        if frame_num < 1 or frame_num > frame_count:
            continue

        label_lines: list[str] = []

        target_list = frame_elem.find("target_list")
        if target_list is None:
            continue

        for target in target_list.findall("target"):
            box = target.find("box")
            attr = target.find("attribute")

            if box is None:
                continue

            vehicle_type = "car"
            if attr is not None:
                vehicle_type = attr.attrib.get("vehicle_type", "car").lower()

            if vehicle_type not in class_map:
                if skip_unknown_classes:
                    continue
                class_id = 0
            else:
                class_id = class_map[vehicle_type]

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            width = float(box.attrib["width"])
            height = float(box.attrib["height"])

            # Clip box to image bounds.
            x1 = max(0.0, left)
            y1 = max(0.0, top)
            x2 = min(float(image_width), left + width)
            y2 = min(float(image_height), top + height)

            clipped_w = x2 - x1
            clipped_h = y2 - y1

            if clipped_w <= 1 or clipped_h <= 1:
                continue

            x_center = (x1 + x2) / 2.0 / image_width
            y_center = (y1 + y2) / 2.0 / image_height
            w_norm = clipped_w / image_width
            h_norm = clipped_h / image_height

            # Safety clamp.
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        label_path = make_label_path(
            labels_dir=labels_dir,
            frame_idx=frame_num,
            label_name_mode=label_name_mode,
            frame_paths=frame_paths,
        )
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def make_label_path(
    labels_dir: Path,
    frame_idx: int,
    label_name_mode: str,
    frame_paths: list[Path] | None = None,
) -> Path:
    """
    frame_index mode:
        000001.txt
        000002.txt

    source_stem mode:
        uses the original frame filename stem.
        For example:
        img00001.jpg -> img00001.txt
    """
    if label_name_mode == "frame_index":
        return labels_dir / f"{frame_idx:06d}.txt"

    if label_name_mode == "source_stem":
        if frame_paths is None:
            raise ValueError("frame_paths required when label_name_mode='source_stem'")
        return labels_dir / f"{frame_paths[frame_idx - 1].stem}.txt"

    raise ValueError("label_name_mode must be 'frame_index' or 'source_stem'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one UA-DETRAC frame sequence + XML annotation to MP4 video + YOLO labels."
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Path to the folder containing one UA-DETRAC frame sequence, e.g. MVI_20011.",
    )
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="Path to the folder containing UA-DETRAC XML files.",
    )
    parser.add_argument(
        "--xml-path",
        default="",
        help="Optional explicit XML path. If omitted, script searches for <frames-dir-name>.xml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where video and labels will be written.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Output video FPS. UA-DETRAC commonly uses 25 FPS.",
    )
    parser.add_argument(
        "--class-mode",
        choices=["coco", "compact"],
        default="coco",
        help="Use COCO class IDs or compact vehicle-only IDs.",
    )
    parser.add_argument(
        "--label-name-mode",
        choices=["frame_index", "source_stem"],
        default="frame_index",
        help="How to name label txt files.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV codec. Use 'mp4v' for .mp4, or 'XVID' for .avi.",
    )

    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    if args.xml_path:
        xml_path = Path(args.xml_path)
    else:
        xml_path = find_matching_xml(frames_dir, annotations_dir)

    sequence_name = frames_dir.name

    output_video = output_dir / "videos" / f"{sequence_name}.mp4"
    labels_dir = output_dir / "labels" / sequence_name

    print(f"[INFO] Frames dir: {frames_dir}")
    print(f"[INFO] XML path:   {xml_path}")
    print(f"[INFO] Video out:  {output_video}")
    print(f"[INFO] Labels out: {labels_dir}")

    frames = collect_frames(frames_dir)

    print(f"[INFO] Found {len(frames)} frames")

    width, height = create_video_from_frames(
        frames=frames,
        output_video=output_video,
        fps=args.fps,
        codec=args.codec,
    )

    print(f"[INFO] Created video with resolution {width}x{height}")

    parse_uadetrac_xml_to_yolo(
        xml_path=xml_path,
        labels_dir=labels_dir,
        frame_count=len(frames),
        image_width=width,
        image_height=height,
        class_mode=args.class_mode,
        label_name_mode=args.label_name_mode,
        frame_paths=frames,
    )

    print("[DONE]")
    print(f"Video:  {output_video}")
    print(f"Labels: {labels_dir}")

    print("\nUse with your benchmark script like this:")
    print(
        f"python run_backend_comparison.py "
        f"--video-source {output_video} "
        f"--labels-dir {labels_dir} "
        f"--output-dir assets/backend_comparison/{sequence_name} "
        f"--roi "
        f"--no-mqtt "
        f"--no-save-outputs"
    )


if __name__ == "__main__":
    main()
