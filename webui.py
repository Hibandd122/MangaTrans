import os
import json
import tempfile
import zipfile
import shutil
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

def process_multiple_images(file_paths, progress=gr.Progress()):
    if not file_paths: return None, "Vui lòng tải lên ít nhất 1 ảnh."
    
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "translated")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        image_files = [f for f in file_paths if f.lower().endswith(valid_exts)]
        
        if not image_files:
            return None, "Không tìm thấy định dạng ảnh hợp lệ trong các file tải lên."
            
        # Sắp xếp file theo tên
        image_files.sort(key=lambda x: os.path.basename(x))
        
        for i, img_path in enumerate(progress.tqdm(image_files, desc="Đang dịch nhiều ảnh...")):
            filename = os.path.basename(img_path)
            out_img_path = os.path.join(output_dir, filename)
            
            try:
                pipeline.process_image(img_path, out_img_path)
            except Exception as e:
                # Fallback: copy ảnh gốc nếu dịch lỗi
                shutil.copy2(img_path, out_img_path)
                print(f"Lỗi khi dịch {img_path}: {e}")
                
        # Nén thư mục kết quả thành file zip
        result_zip = os.path.join(temp_dir, "translated_images.zip")
        shutil.make_archive(result_zip.replace('.zip', ''), 'zip', output_dir)
        
        return result_zip, f"Thành công! Đã dịch {len(image_files)} ảnh."
        
    except Exception as e:
        import traceback
        return None, f"Lỗi:\n{str(e)}\n\n{traceback.format_exc()}"

def process_archive(archive_path, progress=gr.Progress()):
    if not archive_path: return None, "Vui lòng tải lên file nén."
    
    if not os.path.exists(archive_path): return None, "File không tồn tại."
    
    filename = os.path.basename(archive_path)
    is_zip = filename.lower().endswith(('.zip', '.cbz'))
    
    temp_dir = tempfile.mkdtemp()
    extract_dir = os.path.join(temp_dir, "extracted")
    output_dir = os.path.join(temp_dir, "translated")
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Giải nén
        if is_zip:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            try:
                import patoolib
                patoolib.extract_archive(archive_path, outdir=extract_dir)
            except ImportError:
                return None, "Cần cài thư viện patool để giải nén RAR: pip install patool"
            except Exception as e:
                return None, f"Lỗi giải nén (Cần cài đặt unrar/WinRAR trên máy tính/Colab cho định dạng RAR):\n{str(e)}"
                
        # Tìm tất cả file ảnh
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        image_files = []
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    image_files.append(os.path.join(root, file))
                    
        if not image_files:
            return None, "Không tìm thấy ảnh nào trong file nén."
            
        # Sắp xếp file theo tên (ví dụ: 001.jpg, 002.jpg)
        image_files.sort()
        
        # Dịch từng ảnh
        for i, img_path in enumerate(progress.tqdm(image_files, desc="Đang dịch file nén...")):
            rel_path = os.path.relpath(img_path, extract_dir)
            out_img_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            
            try:
                pipeline.process_image(img_path, out_img_path)
            except Exception as e:
                # Fallback: copy ảnh gốc nếu dịch lỗi
                shutil.copy2(img_path, out_img_path)
                print(f"Lỗi khi dịch {img_path}: {e}")
                
        # Nén thư mục kết quả thành file zip
        result_zip = os.path.join(temp_dir, f"translated_{os.path.splitext(filename)[0]}.zip")
        shutil.make_archive(result_zip.replace('.zip', ''), 'zip', output_dir)
        
        return result_zip, f"Thành công! Đã dịch {len(image_files)} ảnh."
        
    except Exception as e:
        import traceback
        return None, f"Lỗi:\n{str(e)}\n\n{traceback.format_exc()}"


with gr.Blocks(title="MangaTrans Web UI") as app:
    gr.Markdown("# MangaTrans Web UI\n*(Detector: comic-text-detector | Inpaint: lama-manga | Dịch: DeepSeek)*")
    
    with gr.Tabs():
        with gr.Tab("Dịch Ảnh Lẻ"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(type="pil", label="Ảnh gốc")
                    run_btn = gr.Button("Dịch Ảnh", variant="primary")
                with gr.Column():
                    output_img = gr.Image(type="pil", label="Ảnh kết quả")
                    output_json = gr.Textbox(label="Dữ liệu JSON", lines=10)
            
            run_btn.click(fn=run_translation, inputs=[input_img], outputs=[output_img, output_json])
            
        with gr.Tab("Dịch Nhiều Ảnh Cùng Lúc"):
            gr.Markdown("Chọn nhiều ảnh cùng một lúc từ máy tính. Hệ thống sẽ tự động dịch tất cả và trả về một file `.zip` chứa các ảnh đã dịch.")
            with gr.Row():
                with gr.Column():
                    multi_input = gr.File(label="Tải lên nhiều ảnh", file_count="multiple", file_types=["image"])
                    multi_btn = gr.Button("Bắt đầu Dịch", variant="primary")
                with gr.Column():
                    multi_output = gr.File(label="Tải về file nén (ZIP)")
                    multi_log = gr.Textbox(label="Trạng thái", lines=5)
                    
            multi_btn.click(fn=process_multiple_images, inputs=[multi_input], outputs=[multi_output, multi_log])
            
        with gr.Tab("Dịch Cả Chương (ZIP/RAR/CBZ/CBR)"):
            gr.Markdown("Nén toàn bộ thư mục ảnh của 1 chương thành file `.zip` hoặc `.cbz` rồi tải lên đây. Quá trình dịch sẽ chạy trên từng ảnh và trả về cho bạn một file nén mới.")
            with gr.Row():
                with gr.Column():
                    archive_input = gr.File(label="Tải lên file nén (ZIP/RAR)", file_types=[".zip", ".rar", ".cbz", ".cbr"])
                    archive_btn = gr.Button("Bắt đầu Dịch Chương", variant="primary")
                with gr.Column():
                    archive_output = gr.File(label="Tải về file đã dịch (ZIP)")
                    archive_log = gr.Textbox(label="Trạng thái", lines=5)
                    
            archive_btn.click(fn=process_archive, inputs=[archive_input], outputs=[archive_output, archive_log])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
