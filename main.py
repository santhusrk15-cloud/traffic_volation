import argparse
from pathlib import Path

from traffic_system.plate_detection import split_yolo_dataset, train_plate_detector_cpu
from traffic_system.report_generator import render_violation_report
from traffic_system.violation_pipeline import TrafficViolationPipeline


def run_inference(args: argparse.Namespace) -> None:
    pipeline = TrafficViolationPipeline(
        helmet_model_path=args.helmet_model,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        save_crops=args.save_crops,
    )
    records = pipeline.process_image(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        save_visuals=args.save_visuals,
    )
    print(render_violation_report(records))


def run_plate_training(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    data_yaml = split_yolo_dataset(dataset_root=dataset_root, val_ratio=args.val_ratio)
    best_model = train_plate_detector_cpu(
        data_yaml_path=data_yaml,
        output_root=Path(args.output_root),
        epochs=args.epochs,
        imgsz=args.imgsz,
    )
    print(f"Training complete. Best model: {best_model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Traffic Violation Detection System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser("infer", help="Run violation inference on image")
    infer_parser.add_argument("--image", required=True, help="Input traffic image path")
    infer_parser.add_argument("--helmet-model", required=True, help="YOLOv8 helmet model path")
    infer_parser.add_argument("--vehicle-model", default="yolov8n.pt", help="YOLO vehicle model path")
    infer_parser.add_argument("--plate-model", required=True, help="YOLOv8 plate model path")
    infer_parser.add_argument("--output-dir", default="outputs", help="Output directory")
    infer_parser.add_argument("--save-visuals", action="store_true", help="Save annotated output image")
    infer_parser.add_argument("--save-crops", action="store_true", help="Save enhanced plate crops")
    infer_parser.set_defaults(func=run_inference)

    train_parser = subparsers.add_parser("train-plates", help="Train YOLOv8n plate detector on CPU")
    train_parser.add_argument("--dataset-root", required=True, help="YOLO dataset root containing images/ and labels/")
    train_parser.add_argument("--output-root", default="runs/plate_training", help="YOLO training project output root")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    train_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    train_parser.set_defaults(func=run_plate_training)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
