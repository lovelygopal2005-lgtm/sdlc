!pip install tranformers torch gradio pyPDF2
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Text generation
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""

    try:
        file_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
        pdf_reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Requirement analysis
def requirement_analysis(pdf_file, prompt_text):
    if pdf_file is not None:
        content = extract_text_from_pdf(pdf_file)
        analysis_prompt = (
            "Analyze the following document and extract key software requirements. "
            "Organize them into functional requirements, non-functional requirements, "
            "and technical requirements.\n\n" + content
        )
    else:
        analysis_prompt = (
            "Analyze the following requirement and organize them into functional requirements, "
            "non-functional requirements, and technical requirements.\n\n" + prompt_text
        )
    return generate_response(analysis_prompt, max_length=1200)

# Code generation
def code_generation(prompt, language):
    code_prompt = f"Generate {language} code for the following requirement:\n\n{prompt}\n\nCode:"
    return generate_response(code_prompt, max_length=1200)

# Gradio app
with gr.Blocks() as app:
    gr.Markdown("# ðŸ¤– AI Code Analysis & Generator")

    with gr.Tabs():
        with gr.TabItem("Code Analysis"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                    prompt_input = gr.Textbox(
                        label="Or write requirements here",
                        placeholder="Describe your software requirements...",
                        lines=5
                    )
                    analyze_btn = gr.Button("Analyze")

                with gr.Column():
                    analysis_output = gr.Textbox(label="Requirements Analysis", lines=20)

            analyze_btn.click(requirement_analysis, inputs=[pdf_upload, prompt_input], outputs=analysis_output)

        with gr.TabItem("Code Generation"):
            with gr.Row():
                with gr.Column():
                    code_prompt_input = gr.Textbox(label="Code Requirements")
                    language_input = gr.Textbox(
                        label="Programming Language",
                        placeholder="e.g., Python, Java, JavaScript"
                    )
                    generate_btn = gr.Button("Generate Code")

                with gr.Column():
                    code_output = gr.Textbox(label="Generated Code", lines=20)

            generate_btn.click(code_generation, inputs=[code_prompt_input, language_input], outputs=code_output)

app.launch(share=True)

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Core text generation
def generate_response(prompt, max_length=1024):
    if not prompt or prompt.strip() == "":
        return "âš ï¸ Please provide valid input."

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response if response else "âš ï¸ No response generated."

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        return ""
    try:
        file_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file
        pdf_reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else "âš ï¸ No readable text found in PDF."
    except Exception as e:
        return f"â Œ Error reading PDF: {str(e)}"

# SDLC Phase Analysis
def sdlc_analysis(phase, pdf_file, user_input):
    content = ""
    if pdf_file is not None:
        content = extract_text_from_pdf(pdf_file)
    else:
        content = user_input

    if not content:
        return "âš ï¸ Please upload a PDF or enter requirements."

    prompt = f"Perform {phase} phase analysis for the following requirements:\n\n{content}"
    return generate_response(prompt, max_length=1200)

# Code Generation
def code_generation(prompt, language):
    if not language:
        return "âš ï¸ Please specify a programming language."
    code_prompt = f"Generate {language} code for the following requirement:\n\n{prompt}\n\nCode:"
    return generate_response(code_prompt, max_length=1200)

# Gradio App
with gr.Blocks() as app:
    gr.Markdown("# ðŸš€ AI-Enhanced SDLC Assistant")

    with gr.Tabs():
        # Requirement Analysis
        with gr.TabItem("Requirement Analysis"):
            with gr.Row():
                with gr.Column():
                    pdf_upload = gr.File(label="Upload Requirement PDF", file_types=[".pdf"])
                    req_input = gr.Textbox(label="Or enter requirements", lines=5)
                    analyze_btn = gr.Button("Analyze Requirements")
                with gr.Column():
                    req_output = gr.Textbox(label="Requirements Analysis", lines=20)
            analyze_btn.click(lambda pdf, text: sdlc_analysis("Requirement Analysis", pdf, text),
                              inputs=[pdf_upload, req_input], outputs=req_output)

        # System Design
        with gr.TabItem("System Design"):
            with gr.Row():
                with gr.Column():
                    design_input = gr.Textbox(label="Enter system design details", lines=5)
                    design_btn = gr.Button("Generate Design")
                with gr.Column():
                    design_output = gr.Textbox(label="System Design Output", lines=20)
            design_btn.click(lambda text: generate_response(f"Generate a system design for:\n\n{text}", max_length=1200),
                             inputs=design_input, outputs=design_output)

        # Implementation
        with gr.TabItem("Implementation"):
            with gr.Row():
                with gr.Column():
                    impl_input = gr.Textbox(label="Enter implementation details", lines=5)
                    impl_btn = gr.Button("Generate Implementation Plan")
                with gr.Column():
                    impl_output = gr.Textbox(label="Implementation Output", lines=20)
            impl_btn.click(lambda text: generate_response(f"Create an implementation strategy for:\n\n{text}", max_length=1200),
                           inputs=impl_input, outputs=impl_output)

        # Testing
        with gr.TabItem("Testing"):
            with gr.Row():
                with gr.Column():
                    test_input = gr.Textbox(label="Enter testing requirements", lines=5)
                    test_btn = gr.Button("Generate Test Cases")
                with gr.Column():
                    test_output = gr.Textbox(label="Test Plan Output", lines=20)
            test_btn.click(lambda text: generate_response(f"Generate detailed test cases and scenarios for:\n\n{text}", max_length=1200),
                           inputs=test_input, outputs=test_output)

        # Deployment
        with gr.TabItem("Deployment"):
            with gr.Row():
                with gr.Column():
                    dep_input = gr.Textbox(label="Enter deployment requirements", lines=5)
                    dep_btn = gr.Button("Generate Deployment Plan")
                with gr.Column():
                    dep_output = gr.Textbox(label="Deployment Output", lines=20)
            dep_btn.click(lambda text: generate_response(f"Create a deployment plan for:\n\n{text}", max_length=1200),
                          inputs=dep_input, outputs=dep_output)

        # Maintenance
        with gr.TabItem("Maintenance"):
            with gr.Row():
                with gr.Column():
                    maint_input = gr.Textbox(label="Enter maintenance requirements", lines=5)
                    maint_btn = gr.Button("Generate Maintenance Plan")
                with gr.Column():
                    maint_output = gr.Textbox(label="Maintenance Output", lines=20)
            maint_btn.click(lambda text: generate_response(f"Generate a maintenance strategy for:\n\n{text}", max_length=1200),
                            inputs=maint_input, outputs=maint_output)

        # Code Generator
        with gr.TabItem("Code Generator"):
            with gr.Row():
                with gr.Column():
                    code_req = gr.Textbox(label="Code Requirements", lines=5)
                    lang_input = gr.Textbox(label="Programming Language", placeholder="e.g., Python, Java, JavaScript")
                    code_btn = gr.Button("Generate Code")
                with gr.Column():
                    code_out = gr.Textbox(label="Generated Code", lines=20)
            code_btn.click(code_generation, inputs=[code_req, lang_input], outputs=code_out)

app.launch(share=True)
