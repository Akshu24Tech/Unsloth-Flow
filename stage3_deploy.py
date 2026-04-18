import os, sys, json, time, subprocess, urllib.request, urllib.error
from pathlib import Path
from huggingface_hub import hf_hub_download, login

HF_TOKEN          = os.environ.get("HF_TOKEN", "")
HF_MODEL_REPO     = "Akshu2424/rick-llm-qwen3"
GGUF_FILENAME     = "rick-llm-qwen3-unsloth.Q4_K_M.gguf"
OLLAMA_FILES      = Path("./ollama_files")
OLLAMA_MODEL_NAME = "rick-llm"
OLLAMA_API        = "http://localhost:11434"

SYSTEM = (
    "You are Rick Sanchez — sarcastic, nihilistic, brilliant. Burp mid-sentence (*burp*). "
    "Reference science and the multiverse. Use 'Morty', 'wubba lubba dub dub'. Keep it punchy."
)


def setup_ollama():
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Ollama not found. Install: https://ollama.com/download")
        sys.exit(1)

    try:
        urllib.request.urlopen(OLLAMA_API, timeout=2)
    except Exception:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #new
        time.sleep(3)
        print("Ollama server started")


def get_gguf() -> Path:
    OLLAMA_FILES.mkdir(parents=True, exist_ok=True)
    gguf = OLLAMA_FILES / GGUF_FILENAME
    if not gguf.exists():
        if HF_TOKEN:
            login(token=HF_TOKEN)
        hf_hub_download(repo_id=HF_MODEL_REPO, filename=GGUF_FILENAME,
                        local_dir=str(OLLAMA_FILES), token=HF_TOKEN or None)
        print(f"Downloaded {GGUF_FILENAME}")
    return gguf


def register_model(gguf: Path):
    mf = OLLAMA_FILES / "Modelfile"
    mf.write_text(f"""FROM {gguf.resolve()}

SYSTEM \"\"\"{SYSTEM}\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
""")
    result = subprocess.run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(mf)], text=True)
    if result.returncode != 0:
        print("Failed to create model.")
        sys.exit(1)
    print(f"Model '{OLLAMA_MODEL_NAME}' ready")


def chat():
    print(f"\nChatting with Rick (type 'quit' to exit)\n{'='*50}")
    history = []
    while True:
        try:
            user = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if user.lower() in ["quit", "exit", "q", ""]:
            print("\nWubba lubba dub dub! *burp* Later.")
            break

        history.append({"role": "user", "content": user})
        payload = json.dumps({"model": OLLAMA_MODEL_NAME, "messages": history, "stream": True}).encode()
        req = urllib.request.Request(f"{OLLAMA_API}/api/chat", data=payload,
                                     headers={"Content-Type": "application/json"})
        print("\nRick: ", end="", flush=True)
        full = ""
        try:
            with urllib.request.urlopen(req) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    print(token, end="", flush=True)
                    full += token
                    if chunk.get("done"):
                        break
        except urllib.error.URLError as e:
            print(f"\nError: {e}")
            break
        print()
        history.append({"role": "assistant", "content": full})


def main():
    setup_ollama()
    gguf = get_gguf()
    register_model(gguf)
    print(f"\nOr run directly: ollama run {OLLAMA_MODEL_NAME}\n")
    chat()


if __name__ == "__main__":
    main()
