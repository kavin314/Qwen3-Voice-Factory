import os
import sys
import warnings
import logging

# 跳過 SSL 憑證驗證（本機 CA 不在信任鏈時的常見問題）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests
import functools
_orig_request = requests.Session.request
@functools.wraps(_orig_request)
def _no_verify_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = _no_verify_request

# --- SILENCER BLOCK ---
# 1. Suppress Python Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*flash-attn.*")
warnings.filterwarnings("ignore", message=".*SoX.*")

# 2. Filter noisy print() output from third-party libs
import builtins
_orig_print = builtins.print
_PRINT_BLOCKLIST = ("flash-attn", "flash_attn", "SoX", "sox")
def _filtered_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if any(kw in msg for kw in _PRINT_BLOCKLIST):
        return
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        _orig_print(safe, **{k: v for k, v in kwargs.items() if k != "end"})
builtins.print = _filtered_print

# 2. Mute Loggers (show errors only)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# ----------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
import torch
import soundfile as sf
import time
import librosa
import numpy as np
import psutil
from qwen_tts import Qwen3TTSModel

# --- CONFIGURATION ---


OUTPUT_DIR = "outputs_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Definitions
# 0.6B 版本讓 GTX 1060 6GB 跑得動；Designer 無 0.6B 版本，保留 1.7B
MODELS_CONFIG = {
    "director": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "cloner":   "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "designer": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
}

# Global Model Cache
loaded_models = {
    "director": None,
    "cloner": None,
    "designer": None
}
current_loaded_mode = None

# --- SYSTEM MONITOR ---
def get_system_stats():
    """System Monitor (CPU/RAM/VRAM) with HTML Styling."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        
        vram_display = "N/A"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
            used_gb = used / (1024**3)
            total_gb = total / (1024**3)
            percent = (used / total) * 100
            # Change color if VRAM usage is high (>90%)
            color = "#ff4444" if percent > 90 else "#00ff88"
            vram_display = f"<span style='color:{color}'>{used_gb:.1f}GB</span> / {total_gb:.1f}GB ({percent:.0f}%)"
        
        return f"""
        <div style="display: flex; gap: 20px; font-family: 'Consolas', monospace; font-size: 14px; color: #ccc; align-items: center; justify-content: flex-end; height: 100%;">
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🖥️ CPU: {cpu}%
            </div>
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🧠 RAM: {ram}%
            </div>
            <div style="background: #1a1f2e; padding: 5px 10px; border-radius: 6px; border: 1px solid #333;">
                🎮 VRAM: {vram_display}
            </div>
        </div>
        """
    except Exception:
        return "Loading Stats..."

# --- SMART MODEL LOADER ---
def unload_other_models(keep_mode):
    """切換 Tab 時卸載其他模型，釋放 VRAM。"""
    global loaded_models, current_loaded_mode
    for mode, model in loaded_models.items():
        if mode != keep_mode and model is not None:
            print(f"🗑️ Unloading {mode} to free VRAM...")
            del loaded_models[mode]
            loaded_models[mode] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_specific_model(mode):
    global loaded_models, current_loaded_mode
    if loaded_models[mode] is not None:
        return loaded_models[mode]

    unload_other_models(keep_mode=mode)

    model_id = MODELS_CONFIG[mode]
    print(f"⏳ Loading {mode}: {model_id}")

    # 停用 bfloat16 Tensor Op（Pascal 沒有 Tensor Core，否則 cuBLAS 會 EXECUTION_FAILED）
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False

    use_gpu = torch.cuda.is_available()

    def _load_model(hf_kwargs):
        try:
            return Qwen3TTSModel.from_pretrained(model_id, **hf_kwargs)
        except Exception as net_err:
            if "SSL" in str(net_err) or "certificate" in str(net_err) or "SSLError" in str(net_err):
                print(f"⚠️ SSL error, retrying with local_files_only=True ...")
                return Qwen3TTSModel.from_pretrained(model_id, local_files_only=True, **hf_kwargs)
            raise

    try:
        if use_gpu:
            print(f"🚀 嘗試 GPU (bfloat16) 載入 {mode}…")
            gpu_kwargs = dict(
                device_map="cuda",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = _load_model(gpu_kwargs)
            vram = torch.cuda.memory_allocated(0) / 1e9
            print(f"✅ {mode.upper()} loaded on GPU! VRAM: {vram:.2f}GB")
        else:
            raise RuntimeError("no_cuda")

    except Exception as gpu_err:
        err_msg = str(gpu_err)
        if "no_cuda" not in err_msg:
            print(f"⚠️ GPU 載入失敗 ({err_msg[:80]})，退回 CPU fp32…")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            cpu_kwargs = dict(
                device_map="cpu",
                torch_dtype=torch.float32,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model = _load_model(cpu_kwargs)
            print(f"✅ {mode.upper()} loaded on CPU (fp32).")
        except Exception as e:
            print(f"❌ {mode} 載入失敗: {e}")
            return None

    loaded_models[mode] = model
    current_loaded_mode = mode
    return model


# --- ENGINE 1: DIRECTOR ---
def run_director(text, speaker="Ryan", instruction=""):
    if not text or len(text.strip()) == 0: return None, "⚠️ Please enter text first!"

    model = load_specific_model("director")
    if not model: return None, "Load Error"
    
    print(f"🎬 Director: '{text}' | Speaker: {speaker}")
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker, 
            instruct=instruction if instruction else None,
            language="Auto"
        )
        return save_audio(wavs, sr, f"director_{speaker}"), "✅ Done"
    except Exception as e: return handle_error(e)

# --- ENGINE 2: CLONER ---

def _split_sentences(text, max_chars=40):
    """依標點斷句，每段不超過 max_chars 字元。"""
    import re
    # 先依句末標點切分
    parts = re.split(r'(?<=[。！？!?\.…])', text.strip())
    chunks = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) <= max_chars:
            current += part
        else:
            if current:
                chunks.append(current)
            # 單段超過 max_chars 時強制截斷
            while len(part) > max_chars:
                chunks.append(part[:max_chars])
                part = part[max_chars:]
            current = part
    if current:
        chunks.append(current)
    return chunks if chunks else [text]

def _clone_one(model, text, ref_wav, ref_sr, actual_ref_text, use_x_vector, instruct):
    """對單一文字段落呼叫 generate_voice_clone，回傳 (wavs, sr)。"""
    try:
        return model.generate_voice_clone(
            text=text,
            ref_audio=(ref_wav, ref_sr),
            ref_text=actual_ref_text,
            x_vector_only_mode=use_x_vector,
            language="Auto",
            instruct=instruct,
        )
    except TypeError as te:
        if "instruct" in str(te):
            return model.generate_voice_clone(
                text=text,
                ref_audio=(ref_wav, ref_sr),
                ref_text=actual_ref_text,
                x_vector_only_mode=use_x_vector,
                language="Auto",
                prompt=instruct,
            )
        raise

def run_cloner(text, ref_audio, ref_text="", style_instruction=""):
    # 0. 基本檢查
    if not ref_audio:
        return None, "⚠️ No Audio provided!"
    if not text or len(text.strip()) == 0:
        return None, "⚠️ Please enter text first!"

    model = load_specific_model("cloner")
    if not model:
        return None, "Load Error"

    # 1. Load Audio
    try:
        ref_wav, ref_sr = librosa.load(ref_audio, sr=16000, mono=True)
    except Exception as e:
        return None, f"⚠️ Error loading audio file: {e}", None

    # 1.1 限制參考音檔長度（GTX 1060 6GB 吃緊）
    max_ref_seconds = 5.0
    max_len = int(max_ref_seconds * ref_sr)
    if len(ref_wav) > max_len:
        ref_wav = ref_wav[:max_len]

    # 2. 決定 clone 模式
    if ref_text and len(ref_text.strip()) > 0:
        print("👉 High-Quality Mode (ICL) with Transcript")
        use_x_vector = False
        actual_ref_text = ref_text
    else:
        print("👉 Fast Mode (X-Vector)")
        use_x_vector = True
        actual_ref_text = None

    instruct = style_instruction.strip() if style_instruction and style_instruction.strip() else None

    # 3. 長文自動分段（fp32 在 GTX 1060 6GB 下保留 KV cache 空間）
    MAX_CHARS = 60
    chunks = _split_sentences(text, max_chars=MAX_CHARS)
    print(f"🧬 Cloning: {len(chunks)} chunk(s), ref {len(ref_wav)/ref_sr:.1f}s @ {ref_sr}Hz")

    try:
        t_start = time.time()
        all_wavs = []
        final_sr = None
        for i, chunk in enumerate(chunks):
            print(f"  [{i+1}/{len(chunks)}] '{chunk}'")
            wavs, sr = _clone_one(model, chunk, ref_wav, ref_sr, actual_ref_text, use_x_vector, instruct)
            # wavs 可能是 list、tensor 或 ndarray
            w = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
            arr = w.squeeze().cpu().numpy() if hasattr(w, 'cpu') else np.squeeze(np.asarray(w))
            all_wavs.append(arr)
            final_sr = sr
            torch.cuda.empty_cache()

        combined = np.concatenate(all_wavs) if len(all_wavs) > 1 else all_wavs[0]
        elapsed = time.time() - t_start
        return save_audio(combined, final_sr, "clone"), f"✅ Done ({elapsed:.1f}s, {len(chunks)} seg)"
    except Exception as e:
        return handle_error(e)


# --- ENGINE 3: CREATOR ---
def run_designer(text, voice_description, instruction=""):
    if not text or len(text.strip()) == 0: return None, "⚠️ Please enter text first!"
    if not voice_description or len(voice_description.strip()) == 0: return None, "⚠️ Please describe the voice first!"

    model = load_specific_model("designer")
    if not model: return None, "Load Error"
    
    print(f"🎨 Design: '{text}'")
    try:
        final_instruct = instruction
        if not final_instruct or len(final_instruct.strip()) == 0:
            final_instruct = voice_description

        wavs, sr = model.generate_voice_design(
            text=text,
            voice_description=voice_description, 
            instruct=final_instruct,
            language="Auto"
        )
        return save_audio(wavs, sr, "design"), "✅ Done"
    except Exception as e: return handle_error(e)

# --- HELPERS ---
def save_audio(wavs, sr, prefix):
    timestamp = int(time.time())
    # Use absolute path
    filename = f"{prefix}_{timestamp}.wav"
    path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
    
    if isinstance(wavs, list):
        data = wavs[0]
    elif hasattr(wavs, 'cpu'):
        data = wavs.squeeze().cpu().numpy()
    else:
        data = np.squeeze(wavs)

    sf.write(path, data, sr)
    return path

def handle_error(e):
    err_str = str(e)
    print(f"❌ Error: {err_str}")
    import traceback
    traceback.print_exc()

    # CUDA illegal memory access 會汙染整個 context，必須卸載所有模型才能繼續使用
    if "illegal memory access" in err_str or "CUDA error" in err_str:
        print("⚠️ CUDA error detected — unloading all models to reset GPU context...")
        global loaded_models, current_loaded_mode
        for mode in list(loaded_models.keys()):
            loaded_models[mode] = None
        current_loaded_mode = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"❌ CUDA 錯誤，已自動卸載模型。請再試一次（模型將重新載入）。\n{err_str}"

    return None, f"❌ Error: {err_str}"

# --- GUI SETUP ---
custom_css = """
body { background-color: #0b0f19; color: #fff; } 
gradio-app { background: #0b0f19 !important; }
.gen-btn { background: linear-gradient(90deg, #ff9966, #ff5e62); color: white; border: none; font-weight: bold; }
.header-row { align-items: center; margin-bottom: 20px; border-bottom: 1px solid #333; padding-bottom: 10px; }
"""
SPEAKERS = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]

spotify_html = """
<div style="text-align: center; margin-top: 10px;">
    If you find this tool helpful, support me on 
    <a href="https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=5d3AbCKgR3GemCemctb8FA" target="_blank" style="color: #1DB954; font-weight: bold; text-decoration: none;">Spotify</a>.
</div>
"""

with gr.Blocks(title="Qwen3 Voice Factory") as demo:
    
    # --- HEADER ---
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1):
            gr.Markdown("# 🏭 Qwen3 Voice Factory")
            # --- CHANGED: More professional subtitle ---
            gr.Markdown("RTX 50 Series Powered | 3 Engines | Portable")
        
        with gr.Column(scale=1):
            stats_display = gr.HTML(value=get_system_stats())
            timer = gr.Timer(1.0)
            timer.tick(get_system_stats, outputs=stats_display)

    # --- TABS ---
    with gr.Tabs():
        
        # TAB 1: DIRECTOR
        with gr.Tab("🎬 Director (Presets)"):
            with gr.Row():
                with gr.Column():
                    t1_text = gr.Textbox(
                        label="Text", 
                        placeholder="Example: Hello, I am using the director mode.", 
                        lines=2
                    )
                    with gr.Row():
                        t1_speaker = gr.Dropdown(SPEAKERS, value="Ryan", label="Speaker")
                        t1_instr = gr.Textbox(
                            label="Style/Instruction", 
                            placeholder="Optional: e.g. Angry, Whispering, Happy", 
                            lines=1
                        )
                    t1_btn = gr.Button("🔊 GENERATE", elem_classes="gen-btn")
                    t1_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t1_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)
            t1_btn.click(run_director, [t1_text, t1_speaker, t1_instr], [t1_out, t1_stat])

         # TAB 2: CLONER
        with gr.Tab("🧬 Voice Cloner"):
            with gr.Row():
                with gr.Column():
                    t2_text = gr.Textbox(
                        label="Text to speak", 
                        placeholder="Example: This is my cloned voice speaking.", 
                        lines=2
                    )
                    
                    # Simple Audio Input (Mic + Upload)
                    t2_ref = gr.Audio(
                        label="Reference Audio (3-10s)", 
                        type="filepath", 
                        sources=["microphone", "upload"]
                    )
                    # Hint for User
                    gr.Markdown("*Note: To save your recording, use the download button in the audio player after recording.*")
                    
                    t2_ref_text = gr.Textbox(
                        label="Transcript of Audio", 
                        placeholder="Optional: Write exactly what is said in the audio for higher quality.", 
                        lines=1
                    )

                    # ✅ 新增：Style / Instruction
                    t2_style = gr.Textbox(
                        label="Style / Instruction",
                        placeholder="Optional: e.g. calm, whispering, angry, excited",
                        lines=1
                    )

                    t2_btn = gr.Button("🧬 CLONE VOICE", elem_classes="gen-btn")
                    t2_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t2_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)

            # ✅ 多傳一個 t2_style 給後端
            t2_btn.click(run_cloner, [t2_text, t2_ref, t2_ref_text, t2_style], [t2_out, t2_stat])


        # TAB 3: CREATOR
        with gr.Tab("🎨 Voice Creator"):
            with gr.Row():
                with gr.Column():
                    t3_text = gr.Textbox(
                        label="Text to speak", 
                        placeholder="Example: I was created from a text description.", 
                        lines=2
                    )
                    t3_desc = gr.Textbox(
                        label="Voice Description (Who?)", 
                        placeholder="Example: A wise old wizard with a deep, raspy voice", 
                        lines=1
                    )
                    t3_instr = gr.Textbox(
                        label="Style/Performance (How?)", 
                        placeholder="Optional: Speaking slowly, whispering, shouting", 
                        lines=1
                    )
                    t3_btn = gr.Button("🎨 CREATE VOICE", elem_classes="gen-btn")
                    t3_stat = gr.Textbox(label="Status")
                with gr.Column():
                    t3_out = gr.Audio(label="Output")
                    gr.Markdown(spotify_html)
            t3_btn.click(run_designer, [t3_text, t3_desc, t3_instr], [t3_out, t3_stat])

if __name__ == "__main__":
    demo.launch( server_name="0.0.0.0",  # 對外 listen
        server_port=7880,
        inbrowser=False,
        show_error=True,
        css=custom_css)