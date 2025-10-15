import cv2
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoProcessor, AutoModelForCausalLM
import supervision as sv
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from collections import Counter
from transformers import PaliGemmaProcessor,PaliGemmaForConditionalGeneration

class Pipeline:
    def __init__(self, device=torch.device("mps")):
        self.device = device
        self.prompt = None
        self.frameCount = 0
        self.captionHistory = {}
        self.initialiseModels()

    def initialiseModels(self):
        self.detectionModel = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        self.detectionModelProcessor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.captionModel = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-pt-224", torch_dtype=torch.bfloat16).to(self.device).eval()
        self.captionProcessor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-pt-224", use_fast=True)
        self.boxAnnotator = sv.BoxAnnotator()
        self.labelAnnotator = sv.LabelAnnotator()
        self.captionCache = []

    def processVideo(self, videoPath, query, outputPath, captionInterval, captionsFile="captions.txt"):
        """Main function for single pass execution"""
        self.frameCount = 0
        self.prompt = query
        capture = cv2.VideoCapture(videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.tracker = sv.ByteTrack(track_activation_threshold=0.4, lost_track_buffer=int(fps*2), minimum_matching_threshold=0.7, frame_rate=fps)
        videoWriter = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        with open(captionsFile, 'w') as cf:
            with tqdm(total=totalFrames, desc="Processing Video") as bar:
                while capture.isOpened():
                    ret, frame = capture.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timeStamp = self.frameCount/fps
                    # Object detection and tracking
                    with torch.no_grad():
                        detections = self.detectObjects(frame)
                        trackedDetections = self.tracker.update_with_detections(detections)

                    # Caption generation
                    if self.frameCount%int(fps*captionInterval)==0:
                        self.generateRegionAwareCaptions(frame, trackedDetections, timeStamp)

                    # Annotate video frames
                    annotatedFrame = self.annotateFrame(frame, trackedDetections)
                    videoWriter.write(cv2.cvtColor(annotatedFrame, cv2.COLOR_RGB2BGR))
                    self.frameCount+=1
                    bar.update(1)

                # finalise captions
                self.finaliseCaptions(cf)

        capture.release()
        videoWriter.release()
        for _, entry in enumerate(self.captionCache):
            outputLine = f"[{entry['start']}-->{entry['end']}] (Object #{entry['trackerID']}): {entry['text']}"
            print(outputLine)
        return outputPath

    def formatTimeStamp(self, seconds):
        """Convert seconds to HH:MM:SS.mmm"""
        td = timedelta(seconds=seconds)
        totalSeconds = td.total_seconds()
        hours = int(totalSeconds//3600)
        minutes = int((totalSeconds%3600)//60)
        seconds = totalSeconds%60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def finaliseCaptions(self, captionFile):
        """Process and write captions in file"""
        sortedCaptions = sorted(self.captionHistory.items(), key=lambda x: x[1]['start'])
        for trackid, entry in sortedCaptions:
            duration = max(0, entry['end'] - entry['start'])
            start = self.formatTimeStamp(entry['start'])
            end = self.formatTimeStamp(entry['start'] + duration)

            captionsCounter = Counter([cap['text'] for cap in entry['captions']])
            mostFrequentCaption = captionsCounter.most_common(1)
            captionText = mostFrequentCaption[0][0]
            captionLine = f"[{start} --> {end}] (Object #{trackid}): {captionText}"
            self.captionCache.append({"start": start, "end": end, "trackerID": trackid, "text": captionText})
            captionFile.write(captionLine + "\n")

    def generateRegionAwareCaptions(self, frame, detections, timestamp):
        pil_frame = Image.fromarray(frame)
        width, height = pil_frame.size
        prompts = []
        tracker_ids = []
        for idx, (x1, y1, x2, y2) in enumerate(detections.xyxy):
            tracker_id = detections.tracker_id[idx]
            if tracker_id not in self.captionHistory:
                self.captionHistory[tracker_id] = {'start': timestamp, 'end': timestamp, 'captions': []}
            else:
                self.captionHistory[tracker_id]['end'] = timestamp
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 <= x1 or y2 <= y1:
                continue
            loc_x1 = int((x1 / width)*1023)
            loc_y1 = int((y1 / height)*1023)
            loc_x2 = int((x2 / width)*1023)
            loc_y2 = int((y2 / height)*1023)
            prompt = f"<image>caption en <loc{loc_y1:04d}><loc{loc_x1:04d}><loc{loc_y2:04d}><loc{loc_x2:04d}>"
            prompts.append(prompt)
            tracker_ids.append(tracker_id)
        
        with torch.no_grad():
            images = [[pil_frame] for _ in prompts]
            inputs = self.captionProcessor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.device)
            outputs = self.captionModel.generate(**inputs, max_new_tokens=100, do_sample=False)
            input_len = inputs["input_ids"].shape[-1]
            decoded = [self.captionProcessor.decode(output[input_len:], skip_special_tokens=True) for output in outputs]
            for tracker_id, caption in zip(tracker_ids, decoded):
                self.captionHistory[tracker_id]['captions'].append({'time': timestamp, 'text': caption})

    def detectObjects(self, frame):
        """Object detection on a frame"""
        pil_frame = Image.fromarray(frame)
        inputs = self.detectionModelProcessor(text=[self.prompt], images=pil_frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detectionModel(**inputs)
        target_sizes = torch.tensor([pil_frame.size[::-1]]).to(self.device)
        results = self.detectionModelProcessor.post_process_object_detection(outputs=outputs, threshold=0.4, target_sizes=target_sizes)[0]
        return sv.Detections.from_transformers(results)

    def annotateFrame(self, frame, detections):
        """Draw annotations on frame"""
        labels = [f"#{tid} {self.prompt[cid]} {conf:.2f}" for tid, cid, conf in zip(detections.tracker_id, detections.class_id, detections.confidence)]
        annotated = self.boxAnnotator.annotate(scene=frame.copy(), detections=detections)
        annotated = self.labelAnnotator.annotate(scene=annotated, detections=detections, labels=labels)
        return annotated

    def run(self, videoPath, queryText, captionInterval=5):
        outputPath = "annotated.mp4"
        captionsFile = "annotatedCaptions.txt"
        queryList = [q.strip() for q in queryText.split(",")]
        processedVideo = self.processVideo(self, videoPath, queryList, outputPath, captionInterval, captionsFile)
        captions = "\n".join(f"[{entry['start']} --> {entry['end']}] (Object #{entry['trackerID']}) {entry['text']}" for entry in self.captionCache)
        return outputPath, captionsFile, captions
