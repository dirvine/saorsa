#!/usr/bin/env python3
"""
Object Detection Module for Saorse Robot System

This module provides real-time object detection using local computer vision models
optimized for Mac M3 with Metal Performance Shaders (MPS).
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from rich.console import Console
from PIL import Image

# Local imports
from model_manager import ModelManager
from mps_optimizer import MPSOptimizer

console = Console()
logger = logging.getLogger(__name__)


class Detection(NamedTuple):
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]  # center x, y
    area: float


@dataclass
class DetectionConfig:
    """Object detection configuration."""
    model_name: str = "facebook/detr-resnet-50"
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.5
    max_detections: int = 100
    input_size: Tuple[int, int] = (800, 600)
    enable_tracking: bool = True
    tracking_max_age: int = 30
    device: str = "auto"  # auto, mps, cpu
    enable_mps_optimization: bool = True
    cache_dir: str = "models/vision"


@dataclass
class TrackingState:
    """Object tracking state."""
    track_id: int
    last_detection: Detection
    age: int = 0
    confirmed: bool = False
    confidence_history: List[float] = field(default_factory=list)


class ObjectDetector:
    """
    Real-time object detection using local vision models.
    
    Supports multiple model architectures optimized for Mac M3:
    - DETR (Detection Transformer)
    - YOLO variants
    - Vision Transformers
    """
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        
        # Model components
        self.model = None
        self.processor = None
        self.feature_extractor = None
        self.device = None
        
        # COCO class names
        self.class_names = self._load_coco_classes()
        
        # Detection state
        self.is_initialized = False
        self.detection_history: List[List[Detection]] = []
        self.tracking_states: Dict[int, TrackingState] = {}
        self.next_track_id = 1
        
        # Performance tracking
        self.fps_counter = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processing_times = []
        
        # Initialize model manager and optimizer
        self.model_manager = ModelManager()
        self.mps_optimizer = MPSOptimizer() if self.config.enable_mps_optimization else None
        
    def _load_coco_classes(self) -> List[str]:
        """Load COCO class names."""
        # Standard COCO classes
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        return coco_classes
        
    async def initialize(self) -> bool:
        """Initialize the object detection model."""
        
        if self.is_initialized:
            return True
            
        try:
            console.print(f"[blue]Initializing object detector with {self.config.model_name}...[/blue]")
            
            # Determine device
            self.device = self._get_optimal_device()
            console.print(f"[blue]Using device: {self.device}[/blue]")
            
            # Load model based on type
            if "detr" in self.config.model_name.lower():
                success = await self._load_detr_model()
            elif "yolo" in self.config.model_name.lower():
                success = await self._load_yolo_model()
            else:
                success = await self._load_transformers_model()
                
            if not success:
                return False
                
            # Optimize for MPS if available
            if self.mps_optimizer and self.device == "mps":
                self.model = self.mps_optimizer.optimize_model(self.model)
                
            # Warm up model
            await self._warmup_model()
            
            self.is_initialized = True
            console.print("[green]✓ Object detector initialized[/green]")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize object detector: {e}")
            console.print(f"[red]✗ Object detector initialization failed: {e}[/red]")
            return False
            
    def _get_optimal_device(self) -> str:
        """Get optimal device for inference."""
        
        if self.config.device != "auto":
            return self.config.device
            
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
            
    async def _load_detr_model(self) -> bool:
        """Load DETR (Detection Transformer) model."""
        
        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            
            # Download/load model
            model_path = await self.model_manager.ensure_model_available(
                self.config.model_name,
                model_type="transformers"
            )
            
            # Load processor and model
            self.processor = DetrImageProcessor.from_pretrained(model_path)
            self.model = DetrForObjectDetection.from_pretrained(model_path)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DETR model: {e}")
            return False
            
    async def _load_yolo_model(self) -> bool:
        """Load YOLO model (using ultralytics)."""
        
        try:
            # Try to import ultralytics YOLO
            try:
                from ultralytics import YOLO
                
                # Map model names
                yolo_models = {
                    "yolov8n": "yolov8n.pt",
                    "yolov8s": "yolov8s.pt", 
                    "yolov8m": "yolov8m.pt",
                    "yolov8l": "yolov8l.pt",
                    "yolov8x": "yolov8x.pt"
                }
                
                model_file = yolo_models.get(self.config.model_name, "yolov8n.pt")
                self.model = YOLO(model_file)
                
                return True
                
            except ImportError:
                logger.warning("Ultralytics YOLO not available, falling back to torch hub")
                
                # Fallback to torch hub YOLO
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.model.to(self.device)
                self.model.eval()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
            
    async def _load_transformers_model(self) -> bool:
        """Load generic transformers model."""
        
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            
            # Download/load model
            model_path = await self.model_manager.ensure_model_available(
                self.config.model_name,
                model_type="transformers"
            )
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForObjectDetection.from_pretrained(model_path)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load transformers model: {e}")
            return False
            
    async def _warmup_model(self):
        """Warm up model with dummy input."""
        
        try:
            console.print("[blue]Warming up model...[/blue]")
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Run inference once to warm up
            await self.detect_objects_async(dummy_image)
            
            console.print("[green]✓ Model warmed up[/green]")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            
    async def detect_objects_async(self, frame: np.ndarray) -> List[Detection]:
        """Asynchronously detect objects in frame."""
        
        if not self.is_initialized:
            logger.warning("Object detector not initialized")
            return []
            
        start_time = time.time()
        
        try:
            # Convert frame to PIL Image
            if frame.dtype == np.uint8:
                image = Image.fromarray(frame)
            else:
                # Convert float to uint8
                frame_uint8 = (frame * 255).astype(np.uint8)
                image = Image.fromarray(frame_uint8)
                
            # Run detection based on model type
            if "detr" in self.config.model_name.lower():
                detections = await self._detect_with_detr(image)
            elif "yolo" in self.config.model_name.lower():
                detections = await self._detect_with_yolo(image)
            else:
                detections = await self._detect_with_transformers(image)
                
            # Filter by confidence
            detections = [d for d in detections if d.confidence >= self.config.confidence_threshold]
            
            # Apply NMS
            detections = self._apply_nms(detections)
            
            # Limit detections
            detections = detections[:self.config.max_detections]
            
            # Update tracking if enabled
            if self.config.enable_tracking:
                detections = self._update_tracking(detections)
                
            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]
                
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps_counter = self.frame_count
                self.frame_count = 0
                self.last_fps_time = current_time
                
            # Store in history
            self.detection_history.append(detections)
            if len(self.detection_history) > 100:
                self.detection_history = self.detection_history[-100:]
                
            return detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
            
    async def _detect_with_detr(self, image: Image.Image) -> List[Detection]:
        """Detect objects using DETR model."""
        
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.1
            )[0]
            
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score.item() < 0.1:
                    continue
                    
                # Convert box coordinates
                x, y, x2, y2 = box.cpu().numpy()
                width = x2 - x
                height = y2 - y
                center_x = int(x + width / 2)
                center_y = int(y + height / 2)
                
                detection = Detection(
                    class_id=label.item(),
                    class_name=self._get_class_name(label.item()),
                    confidence=score.item(),
                    bbox=(int(x), int(y), int(width), int(height)),
                    center=(center_x, center_y),
                    area=width * height
                )
                detections.append(detection)
                
            return detections
            
        except Exception as e:
            logger.error(f"DETR detection failed: {e}")
            return []
            
    async def _detect_with_yolo(self, image: Image.Image) -> List[Detection]:
        """Detect objects using YOLO model."""
        
        try:
            # Convert PIL to numpy for YOLO
            image_np = np.array(image)
            
            # Run inference
            results = self.model(image_np)
            
            detections = []
            
            # Handle ultralytics YOLO format
            if hasattr(results, 'pandas'):
                # Ultralytics format
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            
                            width = x2 - x1
                            height = y2 - y1
                            center_x = int(x1 + width / 2)
                            center_y = int(y1 + height / 2)
                            
                            detection = Detection(
                                class_id=class_id,
                                class_name=self._get_class_name(class_id),
                                confidence=confidence,
                                bbox=(int(x1), int(y1), int(width), int(height)),
                                center=(center_x, center_y),
                                area=width * height
                            )
                            detections.append(detection)
            else:
                # Torch hub YOLO format
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    confidence = row['confidence']
                    class_name = row['name']
                    
                    # Find class ID
                    class_id = 0
                    for i, name in enumerate(self.class_names):
                        if name == class_name:
                            class_id = i
                            break
                            
                    width = x2 - x1
                    height = y2 - y1
                    center_x = int(x1 + width / 2)
                    center_y = int(y1 + height / 2)
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(width), int(height)),
                        center=(center_x, center_y),
                        area=width * height
                    )
                    detections.append(detection)
                    
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
            
    async def _detect_with_transformers(self, image: Image.Image) -> List[Detection]:
        """Detect objects using generic transformers model."""
        
        # This is a fallback implementation - specific models may need custom handling
        return await self._detect_with_detr(image)
        
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"
        
    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        
        if not detections:
            return detections
            
        # Convert to format for NMS
        boxes = []
        scores = []
        for det in detections:
            x, y, w, h = det.bbox
            boxes.append([x, y, x + w, y + h])
            scores.append(det.confidence)
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Apply NMS
        keep_indices = torch.ops.torchvision.nms(boxes, scores, self.config.nms_threshold)
        
        # Return filtered detections
        return [detections[i] for i in keep_indices]
        
    def _update_tracking(self, detections: List[Detection]) -> List[Detection]:
        """Update object tracking across frames."""
        
        # Simple tracking implementation - could be enhanced with Kalman filters
        current_time = time.time()
        
        # Age existing tracks
        for track_id in list(self.tracking_states.keys()):
            track = self.tracking_states[track_id]
            track.age += 1
            
            # Remove old tracks
            if track.age > self.config.tracking_max_age:
                del self.tracking_states[track_id]
                
        # Match detections to existing tracks
        matched_tracks = set()
        updated_detections = []
        
        for detection in detections:
            best_match = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.tracking_states.items():
                if track_id in matched_tracks:
                    continue
                    
                # Calculate distance between detection and track
                dx = detection.center[0] - track.last_detection.center[0]
                dy = detection.center[1] - track.last_detection.center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Check if same class and reasonable distance
                if (detection.class_id == track.last_detection.class_id and 
                    distance < 100 and distance < best_distance):
                    best_match = track_id
                    best_distance = distance
                    
            if best_match is not None:
                # Update existing track
                track = self.tracking_states[best_match]
                track.last_detection = detection
                track.age = 0
                track.confidence_history.append(detection.confidence)
                if len(track.confidence_history) > 10:
                    track.confidence_history = track.confidence_history[-10:]
                track.confirmed = len(track.confidence_history) >= 3
                
                matched_tracks.add(best_match)
                updated_detections.append(detection)
            else:
                # Create new track
                track = TrackingState(
                    track_id=self.next_track_id,
                    last_detection=detection,
                    confidence_history=[detection.confidence]
                )
                self.tracking_states[self.next_track_id] = track
                self.next_track_id += 1
                
                updated_detections.append(detection)
                
        return updated_detections
        
    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """Synchronous wrapper for object detection."""
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                task = loop.create_task(self.detect_objects_async(frame))
                return []  # Return empty for now, result will be available via callback
            else:
                return loop.run_until_complete(self.detect_objects_async(frame))
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.detect_objects_async(frame))
            
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics."""
        
        stats = {
            "fps": self.fps_counter,
            "total_frames": len(self.detection_history),
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "active_tracks": len(self.tracking_states),
            "model_name": self.config.model_name,
            "device": self.device,
            "is_initialized": self.is_initialized
        }
        
        if self.detection_history:
            recent_detections = self.detection_history[-10:]
            avg_detections = np.mean([len(dets) for dets in recent_detections])
            stats["avg_detections_per_frame"] = avg_detections
            
            # Class distribution
            class_counts = {}
            for detections in recent_detections:
                for det in detections:
                    class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            stats["recent_class_distribution"] = class_counts
            
        return stats
        
    def visualize_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Visualize detections on frame."""
        
        frame_vis = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            color = self._get_class_color(detection.class_id)
            cv2.rectangle(frame_vis, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(frame_vis, detection.center, 3, color, -1)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Background for text
            cv2.rectangle(frame_vis, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(frame_vis, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return frame_vis
        
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class visualization."""
        
        # Generate consistent colors for classes
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
        
    def cleanup(self):
        """Clean up resources."""
        
        try:
            if self.model is not None:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                
            if self.processor is not None:
                del self.processor
                
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clear MPS cache if using MPS
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            self.is_initialized = False
            console.print("[yellow]Object detector cleaned up[/yellow]")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Example usage of ObjectDetector."""
    
    console.print("[blue]Saorse Object Detector Demo[/blue]\n")
    
    # Test with different models
    models_to_test = [
        "facebook/detr-resnet-50",
        # "yolov8n",  # Uncomment if ultralytics is available
    ]
    
    for model_name in models_to_test:
        console.print(f"[blue]Testing {model_name}...[/blue]")
        
        config = DetectionConfig(
            model_name=model_name,
            confidence_threshold=0.5,
            input_size=(640, 480)
        )
        
        detector = ObjectDetector(config)
        
        if await detector.initialize():
            # Test with dummy image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some synthetic objects for testing
            cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.circle(test_image, (400, 300), 50, (0, 255, 0), -1)
            
            console.print(f"[blue]Running detection on test image...[/blue]")
            
            start_time = time.time()
            detections = await detector.detect_objects_async(test_image)
            processing_time = time.time() - start_time
            
            console.print(f"[green]✓ Detected {len(detections)} objects in {processing_time:.3f}s[/green]")
            
            for det in detections:
                console.print(f"  {det.class_name}: {det.confidence:.3f} at {det.bbox}")
                
            # Get stats
            stats = detector.get_detection_stats()
            console.print(f"[blue]Stats: {stats['fps']} FPS, {stats['avg_processing_time']:.3f}s avg[/blue]")
            
            # Cleanup
            detector.cleanup()
        else:
            console.print(f"[red]✗ Failed to initialize {model_name}[/red]")
            
        console.print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())