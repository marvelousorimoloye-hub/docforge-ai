#document_parser.py
import gc
import io
import fitz  # PyMuPDF
from pathlib import Path
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.base_models import InputFormat, DocumentStream

def get_optimized_pipeline():
    """Returns a corrected pipeline configured for speed and RAM efficiency."""
    pipeline_options = PdfPipelineOptions()
    
    # 1. Setup RapidOCR (Paddle-based) correctly
    # RapidOcrOptions focuses on model paths/runtime; 
    # we control OCR behavior via pipeline_options.
    pipeline_options.ocr_options = RapidOcrOptions() 
    
    # 2. Trigger OCR behavior globally
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True 
    
    # 3. Speed & Quality Optimizations
    pipeline_options.images_scale = 1.2
    pipeline_options.generate_page_images = False
    
    return pipeline_options


def process_chunk(chunk_data):
    """Worker function to process a single chunk of the PDF."""
    start, end, file_path, stem = chunk_data
    
    try:
        # 1. Extract chunk to memory
        with fitz.open(file_path) as full_doc:
            temp_pdf = fitz.open()
            temp_pdf.insert_pdf(full_doc, from_page=start, to_page=end-1)
            pdf_bytes = temp_pdf.tobytes()
            temp_pdf.close()

        # 2. Setup Stream
        source = DocumentStream(
            name=f"{stem}_chunk_{start}.pdf",
            stream=io.BytesIO(pdf_bytes)
        )
        
        # 3. Initialize converter with optimized options
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=get_optimized_pipeline())
            }
        )
        
        # 4. Convert and cleanup
        result = converter.convert(source)
        md_content = result.document.export_to_markdown()
        
        del pdf_bytes
        gc.collect()
        return md_content

    except Exception as e:
        return f"\n> [Error parsing pages {start+1}-{end}: {str(e)}]\n"

def parse_document(file_path: str):
    file_path = Path(file_path)
    
    with fitz.open(file_path) as doc:
        total_pages = len(doc)
    
    st.info(f"🚀 Parallel Processing {total_pages} pages...")
    progress_bar = st.progress(0)
    
    chunk_size = 4
    # Prepare chunks
    chunks = [
        (start, min(start + chunk_size, total_pages), file_path, file_path.stem)
        for start in range(0, total_pages, chunk_size)
    ]
    
    output_markdown = []
    
    # Use ThreadPoolExecutor for parallel processing
    # max_workers=2 is safer for RAM. If you have 16GB+ RAM, try 3 or 4.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        for i, future in enumerate(futures):
            output_markdown.append(future.result())
            # Update progress
            progress = (i + 1) / len(futures)
            progress_bar.progress(progress)

    st.success(f"✅ Finished: {file_path.name}")
    final_markdown = "\n\n".join(output_markdown)
    
    return {
        "markdown": final_markdown,
        "text": final_markdown,
        "filename": file_path.name
    }
