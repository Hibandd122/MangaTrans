import os
import json
import tempfile
import gradio as gr
from PIL import Image

from mangatrans.pipeline import MangaPipeline
from mangatrans.config import (
    PipelineConfig, DetectorConfig, InpaintConfig, 
    LocalLLMConfig, TranslateConfig
)
from mangatrans.ocr_router import OCRRouterConfig

# Khởi tạo Pipeline Cố Định một lần duy nhất
cfg = PipelineConfig(
    detector=DetectorConfig(backend="comic-text-detector"),
    inpaint=InpaintConfig(backend="lama-manga"),
    translate=TranslateConfig(backend="openrouter", model="deepseek/deepseek-chat"),
    local_llm=LocalLLMConfig(),
    ocr_router=OCRRouterConfig()
)
pipeline = MangaPipeline(cfg, base_dir=".")

def run_translation(image: Image.Image):
    if image is None: return None, "No image provided."
    
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.png")
    output_path = os.path.join(temp_dir, "output.png")
    json_path = os.path.join(temp_dir, "output.json")
    
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(input_path)
    
    try:
        pipeline.process_image(input_path, output_path)
        output_image = Image.open(output_path)
        
        json_data = "No translation data found."
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.dumps(json.load(f), indent=2, ensure_ascii=False)
                
        return output_image, json_data
    except Exception as e:
        import traceback
        return None, f"Error:\n{str(e)}\n\n{traceback.format_exc()}"

with gr.Blocks(title="MangaTrans Web UI") as app:
    gr.Markdown("# MangaTrans Web UI\n*(Detector: comic-text-detector | Inpaint: lama-manga | Dịch: DeepSeek)*")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Ảnh gốc")
            run_btn = gr.Button("Dịch Ảnh", variant="primary")
        with gr.Column():
            output_img = gr.Image(type="pil", label="Ảnh kết quả")
            output_json = gr.Textbox(label="Dữ liệu JSON", lines=10)

    run_btn.click(fn=run_translation, inputs=[input_img], outputs=[output_img, output_json])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
