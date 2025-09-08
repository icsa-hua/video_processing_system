# Task Manager for VPS/OBS 



## TODO 
* [x] Fix: Autobackbone model issue, tensor to Results 
* [x] Add: Cropping of inference image to cars 
* [] Add: Concurrent inference of 360 camera frames 
* [] Add: Jetson docker container 
* [x] Fix: Readme.md Update 
* [] Add: Verification of resizing and merging images 
* [x] Add: Integrate compressed models inside the main pipeline
* [x] Test: MQTT broker from withing the Jetson into the GNT MQTT broker
* [] Fix: Web Interface Connection Error / Model Inference broken pipelin 


## Ideas 
- Integrate DepthAnythingV2 
- ByteTracker into the compressed models. 


## NOTES 
- Compressed integration needs to adjust Results. 
- Integration needs alignment with streaming.py and dummy_app.py 

## Additional Features 
* [x] ADD: Perfomance measurement for modules, and exception catcher. 

## NOTES
- MQTT broker Test: Required a docker image with a test python script that can execute in the 
Jetson NANO environment. The builder for docker must explicitly state the platform for the 
container and then we also need to pass the image through scp as it has credentials. scp 
requires a single hop connection (two jumps) specifying the port and finally, we need to 
ensure that the connection through the docker executes correctly. RC=0 means successfull connection. 


- Tiling Strategy for 360 degrees equirectangular frames 
  Suppose that the input is WxH (e.g. 4096x2048). Tile size matches the model's inference 
  resolution (640x640 for YOLO). Overlap should be around 10-15% to keep objects near tile 
  borders intact. Grid should be: 
  `stride_x = 640 - overlap` and `stride_y = 640 - overlap`. We need to generate (x0,y0) so 
  tiles cover the whole frame; pad with border replication at right/bottom edges if needed. 
  What about Distortion? Because frames are equirectangular, axis-aligned tiles work well around 
  equator but distort near poles. We can stick with rectangular tiles + a bit more vertical overlap 
  near poles or generate _cube-map_ faces (6 respective views) and run YOLO per face; then remap 
  detections back to the equirectangular. 

- Batching should be across frames and tiles, introducing a max_tiles_per_batch. One CUDA 
stream and big batches beat python thread thrash on Nano. Keep model in single process to 
avoid multiple CUDA contexts. 

- LetterBox Math &rarr; Ultralytics already "letterboxes": scales and pads to 640x640 
preserving aspect ratio. For each tile store the scale (sx, sy) from tile->network input, 
pad (px, py) applied by letterbox and offset (x0, y0) tile's origin in the big frame. 
Then reverse after inference and shift back to the original-frame coordinates. 

- Post-Process per frame: cross-tile NMS and feed the tracker. 


- Concurrency model : Proposition through chat. 3-stage bounded queue pipeline; single Cuda model worker
Stage A) Tiler/Preprocessor (CPU) &rarr; Decodes frames, generates tile crops, letterboxes to 640x640 tensors, pushes to `in_queue` 
Stage B) Inference (GPU) &rarr; Pops up to `max_tiles_per_batch`, runs one forward pass, returns raw predictions with a list of tile metadata. 
Stage C) PostProcess (CPU) &rarr; Unletterbox + offset, group by frame_id, per-frame NMS, puch completed frame results downstream. 

[!] Threads can be used for A/C(I/O bound) and a single thread for B (compute bound). Avoid multiprocessing 
on Nano to keep one CUDA context. Make queues small (2-3 batches) to bound memory 

[!] The outer loop still batches 16 original frames. Internally, we are converting those 16 frames into N tiles 
ten running 2-6 inference calls depending on `max_tiles_per_batch` and after stiching, we output 16 reconstructed
Results objects i the original order, and the rest of our pipeline remains unchanged. 


## FAQ 
1. What about overlapped objects (found in multiple tiles)? 
After each tile's prediction, undo letterbox + add tile offset so all boxes are in the original resolution 
Group all boxes for the same frame then run cross-tile NMS (Soft-NMS or Wighted Box Fusion (WBF))
Special 360 seam: if an object straddles the left/right edge, do horizontal wrap: 

2. Is generating coordinates per frame expensive?  
No it isn't, it is rather cheap. It's just a few vectorized ops per box. Even with a few thousand boxes, 
this is <1-2 ms on CPU. The CNN forward pass dominates. 

3. Will my per-frame latency explode? 
With overlap at  10% (64) for the size 640 resolution, stride is 576. 
Tile is 640 and assume 4K equirectangular example 3840x1920 original resolution. 
Columns: ceil((3840-640)/576)+1=6
Rows: 3 => 18 tiles per frame 

If Nano does ~300 ms per 16x640 batch, then 18 tiles leads to 338ms per 4K frame (two calls:16 + 2 tiles)

So yes, per 4K frame latency is roughly the cost of 18 frames of 640x640 inferences which expected
since we are covering more pixels. To improve performance maybe: 

* ROI-gated tiling : Quick downscale pass on the whole frame with high recall, low conf. Only tile at 
full-res around regions that had activity (dets or motion from optical flow) + a margin. With tracking 
predict next ROIs from track boxes (Kalman/extrapolation) and only re-tile there. 

* Dynamic overlap, Seam-aware horizontal strip, TensorRT FP16/INT8, Tune imgsz, short queues and reuse buffers. 

4. Why a separate queue/pipeline? My loop already feeds frames to tiler + model. 
They just decouple rates and help keep the GPU fed if decoding or tiling hiccups. On a Nano, a simple 
synchronous loop is fine. If you ever drop frames or the GPU starves, introduce two tiny ring buffers: (tiler->infer) 
and (infer -> post). Keep capacity 2-3 to bound memory. 

5. How do we run inference multiple times for the same batch but different images? 
The batch 16 frames act the outer unit. Inside, tiles explode the count, so you run multiple micro-batches until all tiles are consumed
Mix tiles from different frames in the same micro-batch to fully utilize `max_tiles_per_batch` 
Keep per-tile `meta = (frame_id, offset, scale, pad, seam_flag, tiles_expected, ...) `
After each forward pass, append predictions to a per-frame accumulator. When `received_tiles == tiles_expected`
for a frame, do cross-tile NMS and emit the Results in original order. 

## Notes LetterBox 
When to use it: 
* if the whole frame is smaller than the model input (e.g. 480p) and we process it full-frame. 
* if we deliberately create non-640 tiles (ROI patches maybe) 
* if we run full-frame at model size (resize + pad) instead of tiling 


#### Is Queue faster than list (Python)
Short-Answer: No 
It's slower because every `put/get` takes locks, condition variables and pickling. Queue is used 
for separating threads and processes. For millisecond budget programs, 
    * Single-Thread &rarr; a plain list is beter 
    * Two threads &rarr; push whole batches, not per-tile items. 
     


#### How to inference the stream when the tile batch is 32 with each frame having multiple tiles? 

