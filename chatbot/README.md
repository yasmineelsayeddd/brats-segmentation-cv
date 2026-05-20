# BraTS Tumor Analysis Chatbot

Demo chatbot for brain tumor segmentation analysis using LangChain + LangGraph with OpenRouter.

## Setup

### 1. Configure API Key

Edit `chatbot/config.yaml` and add your OpenRouter API key:

```yaml
llm:
  api_key: "sk-or-v1-..."
```

### 2. Set Checkpoint Path

Update the model checkpoint path in `chatbot/config.yaml`:

```yaml
ml:
  checkpoint_path: path/to/your/checkpoint.pth
```

### 3. Install Dependencies

**Backend:**
```bash
cd chatbot/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd chatbot/frontend
npm install
```

## Run

**Backend:**
```bash
cd chatbot/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd chatbot/frontend
npm run dev
```

Open http://localhost:5173

## Architecture

```
User → React UI → FastAPI → LangGraph Agent → [Tools] → LLM (OpenRouter)
                                              ↓
                                    segment_scan
                                    analyze_uncertainty
                                    compute_metrics
                                    cascade_detect
                                    explain_findings
```

## Tools

| Tool | Description |
|------|-------------|
| `segment_scan` | Run U-Net segmentation on a 4-channel MRI slice |
| `analyze_uncertainty` | MC-dropout uncertainty analysis |
| `compute_metrics` | Dice, IoU, HD95 against ground truth |
| `cascade_detect` | YOLO detection + U-Net segmentation |
| `explain_findings` | Clinical interpretation of metrics |
