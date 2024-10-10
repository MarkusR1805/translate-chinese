import gradio as gr
import torch
from transformers import pipeline
import webbrowser
import threading
 
# Prüfen, welches System verfügbar ist, ob CUDA, MPS oder CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
# Prüfen, ob MPS (Apple Silicon) verfügbar ist
elif torch.backends.mps.is_available():
    device = torch.device('mps')
# Wenn keines der beiden verfügbar ist, auf CPU zurückgreifen
else:
    device = torch.device('cpu')

print(f"Verwendetes Gerät: {device}")

# Laden der Modelle deutsch / englisch und umgekehrt
de_to_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-de-zh", device=device)
zh_to_de = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-de", device=device)

# Laden der Modelle französisch / deutsch und umgekehrt
en_to_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh", device=device)
zh_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en", device=device)

# Übersetzungsfunktionen
def translate_de_to_zh(text):
    return de_to_zh(text)[0]['translation_text']

def translate_zh_to_de(text):
    return zh_to_de(text)[0]['translation_text']

# Übersetzungsfunktionen französisch / deutsch und umgekehrt
def translate_en_to_zh(text):
    return en_to_zh(text)[0]['translation_text']

def translate_zh_to_en(text):
    return zh_to_en(text)[0]['translation_text']

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Deutsch-Chinesisch und Englisch-Chinesisch Übersetzer und umgekehrt ;-)")
    
    with gr.Row():
        with gr.Column():
            dezh_input = gr.Textbox(label="Deutscher Text", lines=6)
            dezh_output = gr.Textbox(label="Chinesische Übersetzung", lines=6)
            de_to_zh_btn = gr.Button("Deutsch -> Chinesisch")
        
        with gr.Column():
            zhde_input = gr.Textbox(label="Chinesischer Text", lines=6)
            zhde_output = gr.Textbox(label="Deutsche Übersetzung", lines=6)
            zh_to_de_btn = gr.Button("Chinesisch -> Deutsch")
        # Chinesisch / Deutsch  und umgekehrt
    with gr.Row():
        with gr.Column():
            enzh_input = gr.Textbox(label="Englischer Text", lines=6)
            engzh_output = gr.Textbox(label="Chinesische Übersetzung", lines=6)
            en_to_zh_btn = gr.Button("Englisch -> Chinesisch")
        
        with gr.Column():
            zhen_input = gr.Textbox(label="Chinesischer Text", lines=6)
            zhen_output = gr.Textbox(label="Englische Übersetzung", lines=6)
            zh_to_en_btn = gr.Button("Chinesisch -> Englisch")
    
    de_to_zh_btn.click(translate_de_to_zh, inputs=dezh_input, outputs=dezh_output)
    zh_to_de_btn.click(translate_zh_to_de, inputs=zhde_input, outputs=zhde_output)
    # Übersetzungsalgorythmus
    en_to_zh_btn.click(translate_en_to_zh, inputs=enzh_input, outputs=engzh_output)
    zh_to_en_btn.click(translate_zh_to_en, inputs=zhen_input, outputs=zhen_output)

# Funktion zum Öffnen des Browsers
def open_browser():
    webbrowser.open_new("http://127.0.0.1:7860/")

# Starten der Gradio-Oberfläche und Öffnen des Browsers
if __name__ == "__main__":
    # Starten des Browsers in einem separaten Thread
    threading.Timer(1.5, open_browser).start()
    # Starten der Gradio-Oberfläche
    demo.launch()