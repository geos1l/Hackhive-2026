# NeuroRoute

An intelligent voice assistant that dynamically routes queries to the optimal Large Language Model based on real-time benchmarks. Features ESP32 hardware integration for embedded deployment.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Speech-to-Text** | Faster-Whisper (local) |
| **Text-to-Speech** | ElevenLabs API |
| **LLM Router** | Gemini Flash (via OpenRouter) |
| **LLM Agents** | Kimi K2, GPT-OSS 120B, Nemotron 253B, Llama 3.1 405B, Qwen VL |
| **Database** | Supabase (PostgreSQL) |
| **Vision** | OpenCV + Qwen 2.5 VL-72B |
| **Hardware** | ESP32-WROOM-32, ATtiny85, INMP441 Mic, MAX98357A DAC |

---

## Workflow

1. **User presses button** → Microphone activates and records audio
2. **Speech-to-Text** → Faster-Whisper transcribes audio locally
3. **Query Classification** → Gemini analyzes the query and determines its category (reasoning, coding, math, etc.)
4. **Model Selection** → Supabase lookup fetches the best-performing LLM for that category
5. **LLM Response** → Query routed to optimal model (Kimi K2, Llama, Nemotron, etc.)
6. **Text-to-Speech** → ElevenLabs converts response to natural speech
7. **Audio Output** → Speaker plays the response

---

**Built at HackHive 2026**
