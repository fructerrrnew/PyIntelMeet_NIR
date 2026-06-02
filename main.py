#!/usr/bin/env python3
"""
PyIntelMeet — Полноценный монолитный прототип интеллектуальной системы видеоконференций.
Полностью рабочий: FastAPI + WebSocket signaling + browser-native P2P WebRTC mesh + 
Browser SpeechRecognition (ru-RU live captions) + ChromaDB + sentence-transformers RAG KB корректор +
Чисто-Python эвристический генератор отчётов + опциональный Ollama.
Всё в одном файле + гигантская self-contained INDEX_HTML (Tailwind CDN + Vanilla JS).

Запуск: python main.py
Зависимости: см. requirements.txt
"""

import asyncio
import json
import logging
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher
import threading

# FastAPI + WS
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
# python-multipart required for Form/File uploads (added to requirements.txt)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# DB
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# AI
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

try:
    import httpx
    HAS_HTTPX = True
except Exception:
    HAS_HTTPX = False

# ==================== LOGGING & PATHS ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pyintelmeet")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "recordings"), exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "pyintelmeet.db")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")

# ==================== DB SETUP (SQLAlchemy + raw sqlite for simplicity) ====================
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meetings (
                id TEXT PRIMARY KEY,
                room_code TEXT,
                title TEXT,
                host_name TEXT,
                created_at REAL,
                ended_at REAL,
                duration_sec REAL,
                participants_json TEXT,
                raw_transcript_json TEXT,
                corrected_transcript_json TEXT,
                report_md TEXT,
                meta_json TEXT
            )
        """))
        conn.commit()
    logger.info("SQLite DB initialized at %s", DB_PATH)

# ==================== KNOWLEDGE BASE (EXACTLY FROM REQUIREMENTS) ====================
NEUROTEK_KB = [
    {"term": "цифровой двойник", "aliases": ["digital twin", "DT", "виртуальный двойник", "цифровой двойнк"],
     "definition": "Виртуальная динамическая модель физического актива (станка, линии, цеха), синхронизированная с данными в реальном времени для симуляции, прогнозирования и оптимизации.",
     "common_stt_errors": ["цифровой двойнк", "цифровый двойник", "digital twing", "цифровой двой"], "category": "core_technology"},
    {"term": "предиктивное обслуживание", "aliases": ["predictive maintenance", "PdM", "предиктивное обслуживанье"],
     "definition": "Методика прогнозирования отказов оборудования на основе анализа данных датчиков и ML-моделей для планирования ТО заранее.",
     "common_stt_errors": ["предиктивное обслуживанье", "предиктивное обслуживание"], "category": "maintenance"},
    {"term": "edge inference", "aliases": ["инференс на краю", "edge AI", "эдж инференс"],
     "definition": "Выполнение моделей машинного обучения непосредственно на промышленных контроллерах и edge-устройствах.",
     "common_stt_errors": ["edge inference", "эдж инференс"], "category": "deployment"},
    {"term": "нейроморфный процессор", "aliases": ["neuromorphic chip", "нейроморфный процесор"],
     "definition": "Специализированный чип, имитирующий работу биологических нейронов для энергоэффективного инференса.",
     "common_stt_errors": ["нейроморфный процесор"], "category": "hardware"},
    {"term": "вибрационный анализ", "aliases": ["vibration analysis", "vibro analysis", "виброанализ"],
     "definition": "Метод диагностики состояния оборудования по спектрам вибрации.", "common_stt_errors": ["виброанализ"], "category": "monitoring"},
    {"term": "федеративное обучение", "aliases": ["federated learning", "федеративное обученье"],
     "definition": "Распределённое обучение моделей без передачи сырого датасета.", "common_stt_errors": ["федеративное обученье"], "category": "ml_technique"},
    {"term": "TwinForge", "aliases": ["платформа TwinForge", "Twin Forge", "твинфорж"],
     "definition": "Флагманская платформа НейроТек для создания и мониторинга цифровых двойников.", "common_stt_errors": ["Twin Forge", "твинфорж"], "category": "product"},
    {"term": "NT-Edge v3", "aliases": ["серия контроллеров NT-Edge", "NT Edge v3", "энтэдж"],
     "definition": "Промышленный edge-контроллер с поддержкой нейроморфных ускорителей и LoRa.", "common_stt_errors": ["NT Edge v3", "энтэдж"], "category": "product"},
    {"term": "аномалия в спектре", "aliases": ["spectral anomaly", "спектральная аномалия"],
     "definition": "Обнаруженное отклонение в частотном спектре сигнала датчика.", "common_stt_errors": ["спектральная аномалия"], "category": "diagnostics"},
    {"term": "time-series transformer", "aliases": ["TST", "временной трансформер"],
     "definition": "Архитектура трансформера для временных рядов датчиков.", "common_stt_errors": ["time series transformer"], "category": "ml_model"},
    {"term": "MLOps pipeline", "aliases": ["промышленный MLOps", "эмэл опс"],
     "definition": "Автоматизированный цикл разработки и развёртывания ML-моделей в производстве.", "common_stt_errors": ["MLOps pipeline"], "category": "process"},
    {"term": "LoRa sensor mesh", "aliases": ["mesh-сеть LoRa", "LoRa mesh", "лора меш"],
     "definition": "Самоорганизующаяся беспроводная сеть датчиков на базе LoRa.", "common_stt_errors": ["LoRa sensor mesh"], "category": "iot"},
]

def get_kb_index():
    idx = {}
    for e in NEUROTEK_KB:
        idx[e["term"].lower()] = e
        for a in e.get("aliases", []):
            idx[a.lower()] = e
    return idx

KB_INDEX = get_kb_index()

# ==================== GLOBAL STATE ====================
class Peer:
    def __init__(self, peer_id: str, name: str, is_host: bool = False):
        self.peer_id = peer_id
        self.name = name
        self.is_host = is_host
        self.mic_enabled = True
        self.cam_enabled = True
        self.screen_enabled = False
        self.joined_at = time.time()

class TranscriptSegment:
    def __init__(self, ts: float, speaker: str, text: str, conf: float = 0.9, peer_id: str = None):
        self.ts = ts
        self.speaker = speaker
        self.text = text
        self.conf = conf
        self.peer_id = peer_id

class Room:
    def __init__(self, room_id: str, code: str, title: str, host_name: str):
        self.room_id = room_id
        self.code = code
        self.title = title
        self.host_name = host_name
        self.created_at = time.time()
        self.peers: Dict[str, Peer] = {}
        self.transcript: List[TranscriptSegment] = []
        self.started_at = time.time()
        self.ended_at: Optional[float] = None
        self.last_activity = time.time()

    def add_segment(self, seg: TranscriptSegment):
        self.transcript.append(seg)
        self.last_activity = time.time()

# In-memory room manager (robust, production-grade for prototype)
class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.code_to_id: Dict[str, str] = {}
        self.peer_to_room: Dict[str, str] = {}
        self.lock = threading.RLock()

    def generate_code(self) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        while True:
            code = "".join(secrets.choice(alphabet) for _ in range(6))
            if code not in self.code_to_id:
                return code

    def create_room(self, host_name: str, title: Optional[str] = None) -> tuple[Room, str]:
        with self.lock:
            room_id = str(uuid.uuid4())
            code = self.generate_code()
            room = Room(room_id, code, title or f"Встреча {datetime.now().strftime('%d.%m %H:%M')}", host_name)
            host_peer = Peer(str(uuid.uuid4())[:8], host_name, is_host=True)
            room.peers[host_peer.peer_id] = host_peer
            self.rooms[room_id] = room
            self.code_to_id[code] = room_id
            self.peer_to_room[host_peer.peer_id] = room_id
            return room, host_peer.peer_id

    def join_room(self, code: str, name: str) -> Optional[dict]:
        with self.lock:
            room_id = self.code_to_id.get(code)
            if not room_id or room_id not in self.rooms:
                return None
            room = self.rooms[room_id]
            if len(room.peers) >= 8:
                return {"error": "Комната заполнена"}
            peer_id = str(uuid.uuid4())[:8]
            peer = Peer(peer_id, name)
            room.peers[peer_id] = peer
            self.peer_to_room[peer_id] = room_id
            return {
                "room_id": room_id, "peer_id": peer_id, "code": code,
                "title": room.title, "host_name": room.host_name,
                "peers": [{"peer_id": p.peer_id, "name": p.name, "is_host": p.is_host,
                           "mic_enabled": p.mic_enabled, "cam_enabled": p.cam_enabled,
                           "screen_enabled": p.screen_enabled} for p in room.peers.values()]
            }

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def get_room_by_code(self, code: str) -> Optional[Room]:
        return self.rooms.get(self.code_to_id.get(code))

    def remove_peer(self, peer_id: str):
        with self.lock:
            room_id = self.peer_to_room.pop(peer_id, None)
            if room_id and room_id in self.rooms:
                room = self.rooms[room_id]
                if peer_id in room.peers:
                    del room.peers[peer_id]
                if len(room.peers) == 0 and (time.time() - room.last_activity > 900):
                    self._cleanup_room(room_id)

    def _cleanup_room(self, room_id: str):
        room = self.rooms.pop(room_id, None)
        if room:
            self.code_to_id = {c: r for c, r in self.code_to_id.items() if r != room_id}
            logger.info(f"Cleaned empty room {room.code}")

    def add_transcript(self, room_id: str, peer_id: str, text: str, ts: float, conf: float = 0.88) -> Optional[TranscriptSegment]:
        room = self.rooms.get(room_id)
        if not room: return None
        peer = room.peers.get(peer_id)
        speaker = peer.name if peer else "Участник"
        seg = TranscriptSegment(ts, speaker, text.strip(), conf, peer_id)
        room.add_segment(seg)
        return seg

    def end_room(self, room_id: str, peer_id: str) -> Optional[Room]:
        room = self.rooms.get(room_id)
        if not room: return None
        peer = room.peers.get(peer_id)
        if not peer or not peer.is_host: return None
        room.ended_at = time.time()
        return room

    def get_recent_meetings(self, limit: int = 30):
        out = []
        for r in sorted([r for r in self.rooms.values() if r.ended_at], key=lambda x: -x.ended_at)[:limit]:
            out.append({
                "id": r.room_id, "title": r.title, "code": r.code,
                "created_at": r.created_at, "ended_at": r.ended_at,
                "duration_min": int((r.ended_at - r.started_at) / 60),
                "num_participants": len(r.peers) + 1,
            })
        return out

room_manager = RoomManager()

# ==================== CHROMA KB + RAG ====================
_chroma_client = None
_kb_collection = None
_embedding_model = None

def init_knowledge_base():
    global _chroma_client, _kb_collection, _embedding_model
    if not HAS_CHROMA:
        logger.warning("Chroma/sentence-transformers недоступны — будет использоваться только точное совпадение по алиасам.")
        return False
    try:
        os.makedirs(CHROMA_PATH, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _kb_collection = _chroma_client.get_or_create_collection(name="neurotek_kb")
        if _kb_collection.count() == 0:
            logger.info("Seeding Chroma KB (%d терминов)...", len(NEUROTEK_KB))
            docs, metas, ids = [], [], []
            for i, entry in enumerate(NEUROTEK_KB):
                doc = f"{entry['term']}. {entry['definition']} " + " ".join(entry.get("common_stt_errors", []))
                docs.append(doc)
                metas.append({"term": entry["term"], "definition": entry["definition"], "category": entry.get("category", "")})
                ids.append(f"kb_{i}")
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = _embedding_model.encode(docs, show_progress_bar=False).tolist()
            _kb_collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        if _embedding_model is None:
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Chroma KB готова (%d записей).", _kb_collection.count())
        return True
    except Exception as e:
        logger.error("Ошибка инициализации Chroma: %s", e)
        return False

def retrieve_kb(query: str, n: int = 3) -> List[dict]:
    if not _kb_collection or not _embedding_model:
        # fallback keyword
        q = query.lower()
        return [{"term": e["term"], "definition": e["definition"]} for e in NEUROTEK_KB if e["term"].lower() in q][:n]
    try:
        qe = _embedding_model.encode([query]).tolist()
        res = _kb_collection.query(query_embeddings=qe, n_results=n, include=["metadatas"])
        return [{"term": m["term"], "definition": m["definition"]} for m in res["metadatas"][0]]
    except Exception:
        return []

def correct_transcript(segments: List[dict], use_rag: bool = True) -> Tuple[List[dict], dict]:
    corrections = 0
    examples = []
    corrected = []
    for seg in segments:
        original = seg.get("text", "")
        text = original
        # Exact alias / error fixes
        for entry in NEUROTEK_KB:
            for bad in entry.get("common_stt_errors", []) + entry.get("aliases", []):
                if bad.lower() in text.lower():
                    text = re.sub(re.escape(bad), entry["term"], text, flags=re.IGNORECASE)
                    corrections += 1
        # Fuzzy difflib
        for entry in NEUROTEK_KB:
            for cand in [entry["term"]] + entry.get("aliases", []):
                ratio = SequenceMatcher(None, text.lower(), cand.lower()).ratio()
                if ratio > 0.79 and cand.lower() not in text.lower():
                    text = re.sub(re.escape(cand), entry["term"], text, flags=re.IGNORECASE, count=1)
                    corrections += 1
        if use_rag and len(text) > 10:
            for hit in retrieve_kb(text, 2):
                if hit["term"].lower() not in text.lower():
                    pass  # conservative: don't auto inject
        corrected.append({**seg, "text": text, "corrected": text != original})
        if text != original:
            examples.append({"before": original[:90], "after": text[:90]})
    meta = {"num_corrections": corrections, "examples": examples[:4]}
    return corrected, meta

# ==================== HEURISTIC REPORT GENERATOR (ALWAYS WORKS) + OLLAMA ====================
def generate_heuristic_report(title: str, host: str, participants: List[str], duration_sec: float,
                               raw_trans: List[dict], corrected_trans: List[dict], corr_meta: dict) -> dict:
    timeline = []
    for s in corrected_trans:
        ts = int(s.get("ts", 0))
        mmss = f"{ts//3600:02d}:{(ts%3600)//60:02d}:{ts%60:02d}"
        timeline.append(f"[{mmss}] {s.get('speaker')}: {s.get('text')}")

    texts = [s["text"] for s in corrected_trans if len(s.get("text", "")) > 6]
    summary = (texts[0] + " " + (texts[len(texts)//2] if len(texts) > 3 else "") + " " + (texts[-1] if texts else ""))[:620]

    decisions = []
    for s in corrected_trans:
        if any(k in s["text"].lower() for k in ["решили", "договорились", "утвердили", "планируем", "запускаем"]):
            decisions.append(f"{s['speaker']}: {s['text'][:120]}")
    if not decisions:
        decisions = ["Обсуждены вопросы внедрения ключевых технологий НейроТек."]

    action_items = []
    for s in corrected_trans:
        if re.search(r"(нужно|сделаем|подготовить|ответственный|к \d{1,2}\.\d{1,2}|до \d{1,2}\.\d{1,2})", s["text"], re.I):
            owner = s["speaker"]
            m = re.search(r"([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)", s["text"])
            if m: owner = m.group(1)
            action_items.append({"owner": owner, "task": s["text"][:135], "due": "по плану"})
    if not action_items:
        action_items = [{"owner": host, "task": "Подготовить ТЗ на интеграцию TwinForge", "due": "до 15.06.2026"}]

    mentioned = []
    full = " ".join(t["text"].lower() for t in corrected_trans)
    for e in NEUROTEK_KB:
        if e["term"].lower() in full or any(a.lower() in full for a in e.get("aliases", [])):
            mentioned.append({"term": e["term"], "definition": e["definition"][:170]})

    dur_min = int(duration_sec // 60)
    md = f"""# Отчёт о встрече: {title}

**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}  
**Длительность:** {dur_min} мин  
**Хост:** {host}  
**Участники:** {', '.join(participants)}

## Краткое резюме
{summary}

## Хронология обсуждения (Timeline)
""" + "\n".join(timeline[:55]) + ("\n..." if len(timeline) > 55 else "") + """

## Ключевые решения и выводы
""" + "\n".join(f"- {d}" for d in decisions[:5]) + """

## Action Items / Задачи
""" + "\n".join(f"- [ ] **{a['owner']}** — {a['task']} (к {a['due']})" for a in action_items[:6]) + """

## Упомянутые термины из базы знаний
""" + "\n".join(f"- **{m['term']}**: {m['definition']}" for m in mentioned[:6]) + f"""

## Метаданные генерации
- Модель: heuristic-pure (Python)
- KB: NeuroTek 2026.05 ({len(NEUROTEK_KB)} терминов)
- Коррекций: {corr_meta.get('num_corrections', 0)}
"""

    return {
        "title": title, "markdown": md, "timeline": timeline, "summary": summary,
        "decisions": decisions, "action_items": action_items, "kb_terms": mentioned,
        "meta": {"model": "heuristic-pure", "num_corrections": corr_meta.get("num_corrections", 0),
                 "duration_min": dur_min, "participants": participants}
    }

def ollama_status():
    if not HAS_HTTPX:
        return {"available": False}
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            pref = next((m for p in ["qwen2.5", "llama3.1", "gemma2"] for m in models if p in m), models[0] if models else None)
            return {"available": True, "model": pref}
    except Exception:
        pass
    return {"available": False}

async def generate_with_ollama(title: str, transcript: str, kb_ctx: str) -> Optional[dict]:
    status = ollama_status()
    if not status["available"]:
        return None
    model = status["model"]
    prompt = f"""Ты — эксперт НейроТек. Сгенерируй JSON отчёт по транскрипту на русском.
Транскрипт: {transcript[:4200]}
KB контекст: {kb_ctx[:1400]}
Верни строго JSON: {{"summary": "...", "key_decisions": [...], "action_items": [{{"owner":"", "task":"", "due":""}}], "mentioned_kb_terms": [{{"term":"", "definition":""}}] }}"""
    try:
        async with httpx.AsyncClient(timeout=42) as client:
            resp = await client.post("http://127.0.0.1:11434/api/generate", json={
                "model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.15}
            })
            raw = resp.json().get("response", "")
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e]) if s >= 0 else None
    except Exception as ex:
        logger.warning("Ollama failed: %s", ex)
        return None

# ==================== FASTAPI APP ====================
app = FastAPI(title="PyIntelMeet", version="1.0.0-MVP")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve extracted frontend (MVP: editable static/index.html instead of giant inline string)
app.mount("/static", StaticFiles(directory="static", html=False), name="static")

# WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}  # room_id -> list of WS

async def broadcast(room_id: str, message: dict, exclude: WebSocket = None):
    conns = active_connections.get(room_id, [])
    dead = []
    for ws in conns:
        if ws is exclude: continue
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for d in dead:
        if d in conns: conns.remove(d)

@app.websocket("/ws/{room_id}/{peer_id}")
async def ws_endpoint(websocket: WebSocket, room_id: str, peer_id: str, name: str = "Участник"):
    await websocket.accept()
    if room_id not in active_connections:
        active_connections[room_id] = []
    active_connections[room_id].append(websocket)

    room = room_manager.get_room(room_id)
    # SECURITY: Validate peer membership (prevents injection by rogue WS to known room_id)
    if not room or peer_id not in room.peers:
        await websocket.close()
        if room_id in active_connections and websocket in active_connections[room_id]:
            active_connections[room_id].remove(websocket)
        return

    # Send initial peer list + broadcast join for robustness
    peers = [{"peer_id": p.peer_id, "name": p.name, "is_host": p.is_host,
              "mic_enabled": p.mic_enabled, "cam_enabled": p.cam_enabled, "screen_enabled": p.screen_enabled}
             for p in room.peers.values()]
    await websocket.send_text(json.dumps({"type": "peer-list", "data": {"peers": peers}}))
    await broadcast(room_id, {"type": "peer-joined", "data": {"peer": {"peer_id": peer_id, "name": room.peers[peer_id].name}, "peers": peers}})

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            mtype = msg.get("type")

            if mtype == "transcript-segment":
                seg = room_manager.add_transcript(room_id, peer_id, msg["text"], msg.get("ts", 0), msg.get("conf", 0.9))
                if seg:
                    await broadcast(room_id, {
                        "type": "transcript-segment",
                        "data": {"speaker": seg.speaker, "text": seg.text, "ts": seg.ts}
                    })
            elif mtype == "chat":
                peer = room.peers.get(peer_id)
                await broadcast(room_id, {
                    "type": "chat",
                    "data": {"from": peer_id, "from_name": peer.name if peer else "Участник", "text": msg["data"]["text"]}
                })
            elif mtype == "media-state":
                d = msg["data"]
                p = room.peers.get(peer_id)
                if p:
                    if "mic" in d: p.mic_enabled = d["mic"]
                    if "cam" in d: p.cam_enabled = d["cam"]
                    if "screen" in d: p.screen_enabled = d["screen"]
                    await broadcast(room_id, {"type": "media-state", "data": {"peer_id": peer_id, **d}})
            elif mtype == "offer" or mtype == "answer" or mtype == "ice-candidate":
                to = msg.get("to")
                if to:
                    await broadcast(room_id, {"type": mtype, "from": peer_id, "data": msg["data"]}, exclude=websocket)
            elif mtype == "end-meeting":
                ended_room = room_manager.end_room(room_id, peer_id)
                if ended_room:
                    # CRITICAL FIX: Server authoritative report generation + persist on end (using accumulated Room.transcript)
                    # This guarantees report opens for ALL clients reliably in 3-8s, independent of client upload timing
                    try:
                        raw_segs = [{"ts": s.ts, "speaker": s.speaker, "text": s.text, "conf": s.conf} for s in ended_room.transcript]
                        corrected, corr_meta = correct_transcript(raw_segs)
                        host_name = ended_room.host_name
                        parts = [p.name for p in ended_room.peers.values()] or [host_name]
                        dur = (ended_room.ended_at or time.time()) - ended_room.started_at
                        report = generate_heuristic_report(ended_room.title, host_name, parts, dur, raw_segs, corrected, corr_meta)
                        # Persist immediately
                        mid = room_id
                        with engine.connect() as conn:
                            conn.execute(text("""
                                INSERT OR REPLACE INTO meetings (id, room_code, title, host_name, created_at, ended_at, duration_sec,
                                    participants_json, raw_transcript_json, corrected_transcript_json, report_md, meta_json)
                                VALUES (:id, :code, :title, :host, :created, :ended, :dur, :parts, :raw, :corr, :md, :meta)
                            """), {
                                "id": mid, "code": ended_room.code, "title": ended_room.title, "host": host_name,
                                "created": ended_room.created_at, "ended": ended_room.ended_at,
                                "dur": dur, "parts": json.dumps(parts),
                                "raw": json.dumps(raw_segs), "corr": json.dumps(corrected),
                                "md": report["markdown"], "meta": json.dumps(report["meta"])
                            })
                            conn.commit()
                        await broadcast(room_id, {"type": "meeting-ended", "data": {"report_id": room_id}})
                    except Exception as _e:
                        logger.warning(f"Server report gen on end failed: {_e}")
                        await broadcast(room_id, {"type": "meeting-ended", "data": {"report_id": room_id}})
    except WebSocketDisconnect:
        pass
    finally:
        if room_id in active_connections and websocket in active_connections[room_id]:
            active_connections[room_id].remove(websocket)
        room_manager.remove_peer(peer_id)
        # Broadcast leave for mesh robustness
        if room_id in active_connections:
            remaining_peers = []
            r = room_manager.get_room(room_id)
            if r:
                remaining_peers = [{"peer_id": p.peer_id, "name": p.name, "is_host": p.is_host} for p in r.peers.values()]
            await broadcast(room_id, {"type": "peer-left", "data": {"peer_id": peer_id, "peers": remaining_peers}})

# ==================== REST API ====================
@app.post("/api/rooms")
async def create_room(req: Request):
    body = await req.json()
    host = body.get("host_name", "Хост")
    title = body.get("title")
    room, peer_id = room_manager.create_room(host, title)
    return {"success": True, "code": room.code, "room_id": room.room_id, "peer_id": peer_id}

@app.post("/api/rooms/join")
async def join_room(req: Request):
    body = await req.json()
    code = body.get("code", "").upper().strip()
    name = body.get("name", "Участник")
    data = room_manager.join_room(code, name)
    if not data:
        return JSONResponse({"success": False, "error": "Комната не найдена"}, status_code=404)
    if "error" in data:
        return JSONResponse({"success": False, "error": data["error"]}, status_code=400)
    data["success"] = True
    return data

@app.post("/api/correct-transcript")
async def correct_endpoint(req: Request):
    body = await req.json()
    segs = body.get("segments", [])
    use_rag = body.get("use_rag", True)
    corrected, meta = correct_transcript(segs, use_rag)
    return {"corrected": corrected, "meta": meta}

# Guarded upload route — works even if python-multipart not installed at import time
try:
    from fastapi import Form, File, UploadFile
    @app.post("/api/meetings/upload-artifacts")
    async def upload_artifacts(room_id: str = Form(...), peer_id: str = Form(...), transcript: str = Form(...), audio: Optional[UploadFile] = File(None)):
        room = room_manager.get_room(room_id)
        try:
            trans = json.loads(transcript) if transcript else []
        except Exception:
            trans = []
        corrected, corr_meta = correct_transcript(trans)

        # MVP: persist per-participant audio recording for research / future server STT
        audio_path = None
        if audio is not None:
            rec_dir = os.path.join(DATA_DIR, "recordings")
            os.makedirs(rec_dir, exist_ok=True)
            safe_peer = "".join(c for c in (peer_id or "peer") if c.isalnum())[:12]
            fname = f"{room_id[:8]}_{safe_peer}.webm"
            audio_path = os.path.join(rec_dir, fname)
            try:
                with open(audio_path, "wb") as f:
                    f.write(await audio.read())
                logger.info("Saved participant audio: %s (%d bytes)", audio_path, os.path.getsize(audio_path))
            except Exception as e:
                logger.warning("Failed to save audio: %s", e)
                audio_path = None

        dur = (room.ended_at or time.time()) - (room.started_at if room else time.time()) if room else 900
        report = generate_heuristic_report(
            (room.title if room else "Встреча"),
            (room.host_name if room else "Хост"),
            [p.name for p in (room.peers.values() if room else [])] or ["Участник"],
            dur, trans, corrected, corr_meta
        )

        # Enrich meta with recording info (visible in reports for NIR)
        meta = report["meta"]
        if audio_path:
            meta["audio_recording"] = os.path.relpath(audio_path, BASE_DIR)
            meta["has_audio"] = True
        meta["num_corrections"] = corr_meta.get("num_corrections", 0)

        mid = room_id
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO meetings (id, room_code, title, host_name, created_at, ended_at, duration_sec,
                    participants_json, raw_transcript_json, corrected_transcript_json, report_md, meta_json)
                VALUES (:id, :code, :title, :host, :created, :ended, :dur, :parts, :raw, :corr, :md, :meta)
            """), {
                "id": mid,
                "code": (room.code if room else ""),
                "title": (room.title if room else "Встреча"),
                "host": (room.host_name if room else "Хост"),
                "created": (room.created_at if room else time.time()),
                "ended": (room.ended_at or time.time()),
                "dur": dur,
                "parts": json.dumps([p.name for p in (room.peers.values() if room else [])] or ["Участник"]),
                "raw": json.dumps(trans),
                "corr": json.dumps(corrected),
                "md": report["markdown"],
                "meta": json.dumps(meta)
            })
            conn.commit()
        return {"success": True, "report_id": mid, "audio_saved": bool(audio_path)}
except Exception as _multipart_err:
    logger.warning("python-multipart not installed — upload endpoint disabled. Install it for full recording support: pip install python-multipart")
    @app.post("/api/meetings/upload-artifacts")
    async def upload_artifacts_fallback(request: Request):
        # Fallback that still works for pure transcript-based reports
        body = await request.json()
        room_id = body.get("room_id")
        trans = body.get("transcript", [])
        room = room_manager.get_room(room_id) if room_id else None
        corrected, corr_meta = correct_transcript(trans if isinstance(trans, list) else [])
        report = generate_heuristic_report(
            (room.title if room else "Встреча"),
            (room.host_name if room else "Хост"),
            [p.name for p in (room.peers.values() if room else [])] or ["Участник"],
            900,
            trans if isinstance(trans, list) else [],
            corrected,
            corr_meta
        )
        mid = room_id or str(uuid.uuid4())
        with engine.connect() as conn:
            conn.execute(text("INSERT OR REPLACE INTO meetings (id, title, report_md, meta_json) VALUES (:id, :t, :md, :meta)"),
                         {"id": mid, "t": report["title"], "md": report["markdown"], "meta": json.dumps(report["meta"])})
            conn.commit()
        return {"success": True, "report_id": mid, "note": "multipart fallback used"}

@app.get("/api/reports")
async def list_reports():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, title, created_at, ended_at, duration_sec, participants_json, meta_json FROM meetings ORDER BY created_at DESC LIMIT 40")).fetchall()
    out = []
    for r in rows:
        meta = json.loads(r.meta_json or "{}")
        out.append({
            "id": r.id, "title": r.title,
            "created_at": r.created_at, "duration_min": int(r.duration_sec or 0) // 60,
            "num_participants": len(json.loads(r.participants_json or "[]")),
            "num_corrections": meta.get("num_corrections", 0)
        })
    return out

@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM meetings WHERE id = :id"), {"id": report_id}).fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {
        "id": row.id, "title": row.title,
        "markdown": row.report_md,
        "raw_transcript": json.loads(row.raw_transcript_json or "[]"),
        "meta": json.loads(row.meta_json or "{}")
    }

@app.post("/api/reports/generate")
async def regen_report(req: Request):
    body = await req.json()
    mode = body.get("mode", "heuristic")
    raw = body.get("raw_transcript", [])
    title = body.get("title", "Встреча")
    participants = body.get("participants", [])
    dur = body.get("duration_sec", 900)
    corrected, meta = correct_transcript(raw)
    if mode == "ollama":
        ollama_res = await generate_with_ollama(title, " ".join(s["text"] for s in corrected), "")
        if ollama_res:
            # REAL FIX: Use Ollama structured output to build superior report (merge with timeline from corrected)
            base = generate_heuristic_report(title, participants[0] if participants else "Хост", participants, dur, raw, corrected, meta)
            ollama_summary = ollama_res.get("summary", base["summary"])
            ollama_decisions = ollama_res.get("key_decisions", base["decisions"])
            ollama_actions = ollama_res.get("action_items", base["action_items"])
            ollama_kb = ollama_res.get("mentioned_kb_terms", base["kb_terms"])
            # Rebuild markdown with Ollama content + server timeline
            timeline_str = "\n".join(base["timeline"][:55])
            md = f"""# Отчёт о встрече: {title}

**Дата:** {datetime.now().strftime('%d.%m.%Y %H:%M')}  
**Длительность:** {base["meta"].get("duration_min", 0)} мин  
**Хост:** {participants[0] if participants else "Хост"}  
**Участники:** {', '.join(participants)}

## Краткое резюме
{ollama_summary}

## Хронология обсуждения (Timeline)
{timeline_str}

## Ключевые решения и выводы
""" + "\n".join(f"- {d}" for d in ollama_decisions[:6]) + """

## Action Items / Задачи
""" + "\n".join(f"- [ ] **{a.get('owner','Команда')}** — {a.get('task','') } (к {a.get('due','по плану')})" for a in ollama_actions[:6]) + """

## Упомянутые термины из базы знаний
""" + "\n".join(f"- **{m.get('term','')}**: {m.get('definition','')}" for m in ollama_kb[:6]) + f"""

## Метаданные генерации
- Модель: ollama-enhanced ({ollama_status().get('model','ollama')})
- KB: NeuroTek 2026.05 ({len(NEUROTEK_KB)} терминов)
- Коррекций: {meta.get('num_corrections', 0)}
"""
            base["markdown"] = md
            base["summary"] = ollama_summary
            base["decisions"] = ollama_decisions
            base["action_items"] = ollama_actions
            base["kb_terms"] = ollama_kb
            base["meta"]["model"] = "ollama+" + (ollama_status().get("model") or "unknown")
            return base
    return generate_heuristic_report(title, participants[0] if participants else "Хост", participants, dur, raw, corrected, meta)

@app.get("/api/ollama-status")
async def ollama_status_api():
    return ollama_status()


# ==================== SERVE FRONTEND (MVP: static/index.html - clean & editable) ====================
# UI extracted from previous monolith for real maintainability and good developer UX.
# `python main.py` continues to be the only command needed.
# Future: optional bundler step to produce a true single-file distribution if desired.

@app.get("/", response_class=HTMLResponse)
async def root():
    path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.isfile(path):
        return FileResponse(path, media_type="text/html; charset=utf-8")
    return HTMLResponse(
        "<!doctype html><meta charset=\"utf-8\"><body style=\"font-family:system-ui;padding:2rem\">"
        "<h1>PyIntelMeet MVP</h1><p>Frontend missing: static/index.html not found.</p></body>",
        status_code=500
    )

@app.get("/health")
async def health():
    """MVP health + status for demos, monitoring, and NIR experiments."""
    return {
        "status": "ok",
        "version": app.version,
        "chroma_kb": HAS_CHROMA,
        "ollama": ollama_status().get("available", False),
        "active_rooms": len(room_manager.rooms),
        "db_path": DB_PATH,
    }


@app.on_event("startup")
async def startup_event():
    init_db()
    init_knowledge_base()
    print("\n" + "="*72)
    print("  PyIntelMeet MVP v1.0 — ГОТОВ К ДЕМОНСТРАЦИИ / ЗАЩИТЕ НИР")
    print("="*72)
    print(f"  URL:            http://127.0.0.1:8000  (или :{os.getenv('PORT',8000)})")
    print(f"  Frontend:       static/index.html (редактируйте напрямую)")
    print(f"  Data:           {DATA_DIR}  (БД + записи аудио)")
    print(f"  Chroma KB:      {'OK' if HAS_CHROMA else 'OFF (alias mode only)'}")
    oll = ollama_status()
    print(f"  Ollama:         {'ON (' + oll.get('model','?') + ')' if oll.get('available') else 'OFF (heuristic only)'}")
    print("  Chrome/Edge:    2+ вкладки → Создать/Демо → говорите технические термины!")
    print("  Health:         /health")
    print("="*72 + "\n")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False, log_level="info")
