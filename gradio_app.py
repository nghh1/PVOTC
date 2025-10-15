import gradio as gr
from Pipeline import Pipeline
import os
import warnings

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
pipeline = Pipeline()

def processVideoGradio(rawVideo, queryText, captionInterval):
    pipeline.captionCache.clear()
    pipeline.captionHistory.clear()
    outputPath="annotated.mp4"
    pipeline.processVideo(videoPath=rawVideo, query=queryText.split(","), outputPath=outputPath, captionInterval=captionInterval, captionsFile="annotatedCaptions.txt")
    captionText = "\n".join(f"[{entry['start']} --> {entry['end']}] (Object #{entry['trackerID']}) {entry['text']}" for entry in pipeline.captionCache)
    return outputPath, captionText

interface = gr.Interface(fn=processVideoGradio, inputs=[gr.Video(label="Input Video"), gr.Textbox(label="Detection query (e.g. dog / cat, person)"), gr.Slider(0, 5.0, step=1.0, value=1.0, label="Caption Interval (seconds)")], outputs=[gr.Video(label="Annotated Output Video"), gr.Textbox(label="Generated Captions")], title='OW-VITC pipeline', description="Upload a video, enter objects to be tracked and generated captions for region objects", flagging_mode="never")
interface.launch()