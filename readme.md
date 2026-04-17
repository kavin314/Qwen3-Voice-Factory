# 🏭 Qwen3 Voice Factory（RTX 50 系列優化版）

一個本地、可攜式的 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) 圖形介面工具。
特別針對 **NVIDIA RTX 50 系列**（CUDA 12.8 / PyTorch Nightly）進行優化，同時也支援前幾代顯卡（3090／4090）。

> **🎯 適合想快速測試這些模型，又不想處理複雜節點圖（ComfyUI）的使用者。**

![Screenshot](screenshot.png)

## 功能特色

- **🎬 導演模式（Director Mode）：** 選擇預設角色（Ryan、Vivian）並提供情境指令（例如「憤怒」、「耳語」）。
- **🧬 語音複製（Voice Cloner）：** 上傳一段短音訊（3–10 秒）即可複製該聲音（支援高品質 ICL 模式）。
- **🎨 語音創作（Voice Creator）：** 透過文字描述從零創造全新的聲音（Voice Design）。
- **📊 即時硬體監控：** 內建即時儀表板，可在生成過程中監看 VRAM／RAM／CPU 使用率。
- **📂 自動儲存：** 會自動建立 `outputs_audio` 資料夾，並以時間戳記保存每次生成的音檔。
- **可攜式設計：** 不會修改 Windows 系統設定，所有內容都集中在單一資料夾內。

## 安裝方式

1. 將此 repository 下載為 ZIP 檔並解壓縮。
2. 雙擊執行 `install.bat`。
   - 此腳本會自動下載獨立的 Python 3.11 環境。
   - 並安裝 PyTorch Nightly（為支援 Blackwell／RTX 50 系列所必需）。
3. 等待安裝完成。

## 使用方式

1. 雙擊執行 `start.bat`。
2. 瀏覽器會自動開啟至 `http://127.0.0.1:7860`。

## 模型說明
首次使用各分頁時，模型會從 HuggingFace 自動下載（每個模型約 4GB），請確保磁碟空間足夠。

## 系統需求

- Windows 10／11
- NVIDIA 顯示卡（建議 12GB 以上 VRAM）
- 網路連線（安裝及下載模型時需要）

## 🔗 致謝與來源

本專案是一個 GUI 封裝工具，目的是讓 **Qwen 團隊** 的優秀成果更容易被一般使用者取用。所有 AI 能力皆由其模型提供。

- **基礎模型：** 由 [Alibaba Cloud／Qwen Team](https://huggingface.co/Qwen) 開發。
- 歡迎到 HuggingFace 和 GitHub 支持他們的原創作品。

## 🤝 支持作者

這是一個免費的開源專案，我不接受捐款。
不過如果您想表達「感謝」，歡迎到 **[Spotify](https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=4AqQE6GcQpKJFeVk6gJ06g)** 看看我的作品。

一個追蹤或一次聆聽，就是對我最好的支持！🎧
