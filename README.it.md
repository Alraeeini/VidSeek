# VidSeek — ricerca semantica nei video per agenti AI

**VidSeek** è una libreria Python sperimentale, progettata per funzionare localmente e basata sul modello `google/embeddinggemma-2` di Google. Permette a un'applicazione di indicizzare file video locali e recuperare clip pertinenti a partire da richieste in linguaggio naturale. I risultati includono percorsi dei file, timestamp, metadati e un punteggio di similarità coseno, consentendo a un agente AI di decidere quali clip esaminare o aprire.

> VidSeek è un progetto indipendente e sperimentale, non un prodotto ufficiale di Google.

## Cosa fa e cosa non fa

- Trasforma video e testo in vettori semantici confrontabili utilizzando EmbeddingGemma 2.
- Suddivide i video in clip con timestamp prima dell'indicizzazione, consentendo di individuare intervalli temporali specifici.
- Memorizza gli embedding e i metadati in un database SQLite locale.
- Espone metodi Python e schemi di funzioni serializzabili in JSON per gli agenti AI.
- Funziona localmente dopo il download delle dipendenze e dei pesi del modello; non richiede un'API di embedding ospitata su un server.

Gli embedding **non sono** trascrizioni, sottotitoli, descrizioni dettagliate o riassunti generati. La versione iniziale serve a recuperare clip pertinenti. Se il tuo agente deve identificare le parole pronunciate esattamente, spiegare una scena o produrre descrizioni complete, puoi integrarlo con un sistema di riconoscimento vocale come Whisper e/o con un modello generativo visivo-linguistico.

## Requisiti

- Python 3.10 o superiore.
- FFmpeg installato, con `ffmpeg` e `ffprobe` disponibili nel `PATH` per la segmentazione con timestamp.
- RAM/VRAM sufficiente per la configurazione del modello scelta. Al primo utilizzo vengono scaricati i pesi del modello.
- Il modello viene caricato tramite Sentence Transformers e Transformers. EmbeddingGemma 2 supporta `bfloat16` o `float32`; questa libreria non forza l'utilizzo di `float16`.

## Installazione dal sorgente

```bash
git clone https://github.com/Alraeeini/VidSeek.git
cd VidSeek
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Se la tua piattaforma richiede una versione specifica di PyTorch, installa prima la versione appropriata utilizzando il [selettore ufficiale di PyTorch](https://pytorch.org/get-started/locally/), quindi installa VidSeek.

## Avvio rapido

```python
from vidseek import VideoEncoder, VideoIndex

encoder = VideoEncoder(device="cpu")  # usa device="cuda" con una configurazione NVIDIA compatibile
index = VideoIndex("./vidseek.sqlite3", encoder=encoder)

# Crea un embedding separato per ogni clip di 20 secondi.
index.add_video(
    "./mio_video.mp4",
    segment_seconds=20,
    metadata={"title": "Il mio video", "project": "demo"},
)

results = index.search("un cane corre in un campo innevato", top_k=5)

for result in results:
    print(result["video_path"])
    print(result["start_seconds"], result["end_seconds"])
    print(result["score"], result["metadata"])
```

Per creare un solo embedding per l'intero video, usa `index.add_video("./mio_video.mp4", segment_seconds=None)`. In questo caso, se non è possibile determinare la durata del video, i risultati potrebbero non fornire timestamp utili per individuare singole scene.

## Utilizzare VidSeek come strumento per agenti AI

```python
from vidseek import VideoAgentTools, VideoIndex

index = VideoIndex("./vidseek.sqlite3")

# Per impostazione predefinita, l'agente può cercare,
# ma non può indicizzare percorsi arbitrari del filesystem.
tools = VideoAgentTools(index)

# Registra questi schemi nel framework agentico compatibile.
schemas = tools.tool_schemas()

# Per abilitare l'indicizzazione da parte dell'agente,
# autorizza esplicitamente una cartella multimediale:
# tools = VideoAgentTools(index, allowed_roots=["./media_autorizzati"])
# schemas = tools.tool_schemas()  # ora include index_video

# Esegui una chiamata restituita dal framework agentico.
result = tools.execute(
    "search_videos",
    {"query": "una persona apre la portiera di un'auto rossa", "top_k": 3},
)
print(result)
```

Funzioni disponibili:

- `search_videos(query, top_k=5)` — ricerca semantica; restituisce clip video ordinate per rilevanza.
- `index_video(video_path, segment_seconds=20, title=None)` — indicizza un file locale autorizzato dall'utente.

**Nota sulla sicurezza delle integrazioni agentiche:** l'indicizzazione avviata dall'agente è disabilitata, a meno che non venga configurato esplicitamente `allowed_roots`. Possono essere indicizzati soltanto i file che si trovano all'interno delle cartelle autorizzate. Un'applicazione destinata alla produzione dovrebbe imporre anche limiti sulla dimensione dei file, sulla durata e sui codec.

## Come funziona la segmentazione

L'impostazione predefinita `segment_seconds=20` utilizza FFmpeg per creare clip video temporanee, calcola un embedding per ciascuna clip e registra i tempi iniziale e finale. Le clip temporanee vengono eliminate dopo l'indicizzazione, mentre gli embedding e i metadati rimangono in SQLite. Quando si reindicizza un percorso, le righe precedenti vengono sostituite soltanto dopo il calcolo riuscito di tutti i nuovi embedding.

EmbeddingGemma 2 campiona i fotogrammi di ogni clip: non analizza ogni fotogramma alla frequenza originale completa. Puoi modificare `VideoEncoder(fps=..., max_frames=...)` per bilanciare velocità e copertura visiva. Segmenti più brevi possono migliorare la precisione temporale, ma aumentano i tempi di indicizzazione e le dimensioni del database.

## Test

I test unitari utilizzano un encoder simulato e non scaricano il modello:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Limiti attuali

- Implementazione sperimentale in fase alpha; non esiste ancora una versione pubblicata su PyPI.
- Supporta file video locali; `VideoIndex.add_video` non accetta URL o flussi video in streaming.
- La ricerca esamina linearmente l'indice SQLite. Per raccolte di grandi dimensioni, è possibile aggiungere un indice vettoriale come FAISS, sqlite-vec, Qdrant o un altro backend appropriato.
- Non include ancora trascrizione, diarizzazione, OCR, tracciamento degli oggetti, riassunti o ragionamento visivo generativo.
- Qualità e prestazioni dipendono dalla versione del modello, dall'hardware, dal contenuto del video, dalla durata dei segmenti e dalle opzioni di campionamento.

## Riferimenti

- [Guida multimodale di EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [Scheda tecnica di EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2?hl=en)
- [Repository del modello su Hugging Face](https://huggingface.co/google/embeddinggemma-2)

## Licenza

VidSeek viene distribuito secondo i termini della licenza MIT. Consulta il file `LICENSE` del repository.
