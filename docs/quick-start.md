# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up API Key

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

## 3. Run the App

```bash
streamlit run app.py
```

## 4. Use the Chatbot

1. **Upload Documents** - Use the sidebar to upload PDFs, images, or text files
2. **Click "Process Documents"** - Wait for processing to complete
3. **Ask Questions** - Type your question in the chat input
4. **View Sources** - Expand "View Sources" to see retrieved context

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "API key not found" | Check your `.env` file |
| "Module not found" | Run `pip install -r requirements.txt` |
| Slow first run | Models are downloading (one-time) |

## Supported Files

- **PDF** - Text, images, and tables extracted
- **Images** - JPG, PNG (processed with OCR + CLIP)
- **Text** - TXT, MD files

---

That's it! 🚀
