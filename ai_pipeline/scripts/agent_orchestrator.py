"""
PropaBridge — Agent Orchestrator  (Phase D3)
=============================================
Architecture:  Intent Router → Tool Dispatcher → Prompt Augmenter → Gemini Generator

                         ┌──────────────────────────────────────────┐
  user_query ──────────▶ │  1. IntentRouter                         │
                         │     PROPERTY_SEARCH | LEGAL_FAQ |        │
                         │     GENERAL_CHAT                         │
                         └─────────────────┬────────────────────────┘
                                           │ intent + query
                         ┌─────────────────▼────────────────────────┐
                         │  2. ToolDispatcher                       │
                         │     PROPERTY_SEARCH / LEGAL_FAQ          │
                         │       → query_helper.retrieve_chunks()   │
                         │         (Vector Search → Firestore)      │
                         │     GENERAL_CHAT                         │
                         │       → no retrieval (direct to Gemini)  │
                         └─────────────────┬────────────────────────┘
                                           │ chunks (or empty)
                         ┌─────────────────▼────────────────────────┐
                         │  3. PromptAugmenter                      │
                         │     system_prompt + context + user_query │
                         └─────────────────┬────────────────────────┘
                                           │ augmented prompt
                         ┌─────────────────▼────────────────────────┐
                         │  4. GeminiGenerator                      │
                         │     gemini-1.5-pro via Vertex AI SDK     │
                         │     initialized with service_account.json│
                         └─────────────────┬────────────────────────┘
                                           │
                                    AgentResponse

Project:  studio-6084005125-75144
Region:   us-central1
Auth:     service_account.json (root directory)

Dependencies:
    pip install google-cloud-aiplatform google-cloud-firestore
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import vertexai
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, GenerationConfig, SafetySetting, HarmCategory, HarmBlockThreshold

# ── Sibling module (must be in same directory or on PYTHONPATH) ───────────────
from query_helper import retrieve_chunks, format_context_for_gemini

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("propabridge.orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    PROJECT_ID           = "studio-6084005125-75144"
    REGION               = "us-central1"
    SERVICE_ACCOUNT_FILE = Path(__file__).parent.parent / "service_account.json"
    GEMINI_MODEL         = "gemini-1.5-pro"

    # RAG retrieval settings
    TOP_K_PROPERTY       = 5   # chunks for PROPERTY_SEARCH
    TOP_K_LEGAL          = 4   # chunks for LEGAL_FAQ
    TOP_K_DEFAULT        = 3

    # Gemini generation settings
    TEMPERATURE          = 0.3   # lower = more factual / grounded
    MAX_OUTPUT_TOKENS    = 1024
    TOP_P                = 0.85

    # Intent routing — keyword signals used before LLM fallback
    PROPERTY_KEYWORDS = [
        "rent", "apartment", "flat", "house", "duplex", "bq", "self-con",
        "bedroom", "2 bed", "3 bed", "1 bed", "let", "lease", "property",
        "neighborhood", "area", "district", "abuja", "kaduna", "maitama",
        "gwarinpa", "kubwa", "lugbe", "barnawa", "asokoro", "wuse", "jabi",
        "gra", "estate", "location", "cheap", "affordable", "price", "budget",
        "how much", "cost", "tier", "premium", "mid-range", "face-me",
        "buy", "purchase", "sell", "landlord", "tenant", "caution fee",
    ]
    LEGAL_KEYWORDS = [
        "c of o", "certificate of occupancy", "deed of assignment", "gazette",
        "stamp duty", "tenancy agreement", "legal", "document", "title",
        "encumbrance", "mortgage", "reit", "governor's consent", "agreement",
        "omo onile", "agent fee", "commission", "lawyer", "law", "act",
        "rights", "eviction", "quit notice", "recovery of premises",
        "land use act", "registration", "verify", "due diligence",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

class Intent(str, Enum):
    PROPERTY_SEARCH = "PROPERTY_SEARCH"   # Neighborhood / rental / location queries
    LEGAL_FAQ       = "LEGAL_FAQ"         # Title docs, tenancy law, due diligence
    GENERAL_CHAT    = "GENERAL_CHAT"      # Greetings, off-topic, clarifications


@dataclass
class IntentResult:
    intent:     Intent
    confidence: float          # 0.0 – 1.0
    city_hint:  Optional[str]  # "Abuja" | "Kaduna" | None
    method:     str            # "keyword" | "llm_fallback"


@dataclass
class AgentResponse:
    answer:          str
    intent:          Intent
    sources_used:    int                   # number of Firestore chunks retrieved
    source_keys:     list[str]             # chunk source_key values for attribution
    city_filter:     Optional[str]
    latency_ms:      float
    model:           str = Config.GEMINI_MODEL
    retrieval_skipped: bool = False        # True for GENERAL_CHAT

    def __str__(self) -> str:
        lines = [
            f"\n{'═'*62}",
            f"  PropaBridge AI Response",
            f"{'═'*62}",
            f"  Intent   : {self.intent.value}",
            f"  Sources  : {self.sources_used} Firestore chunks",
            f"  City     : {self.city_filter or 'All markets'}",
            f"  Latency  : {self.latency_ms:.0f}ms",
            f"{'─'*62}",
            f"{self.answer}",
            f"{'─'*62}",
        ]
        if self.source_keys:
            lines.append(f"  Sources  : {', '.join(self.source_keys)}")
        lines.append(f"{'═'*62}\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INTENT ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Two-tier classifier:
      Tier 1 — Fast keyword scan (zero API cost, ~0ms)
      Tier 2 — Gemini LLM fallback for ambiguous queries
    """

    def __init__(self, model: GenerativeModel):
        self._model = model     # reuse the already-initialised Gemini model

    def route(self, query: str) -> IntentResult:
        q_lower = query.lower()
        city_hint = self._detect_city(q_lower)

        # ── Tier 1: keyword matching ─────────────────────────────────
        prop_hits  = sum(1 for kw in Config.PROPERTY_KEYWORDS if kw in q_lower)
        legal_hits = sum(1 for kw in Config.LEGAL_KEYWORDS    if kw in q_lower)

        if prop_hits > 0 or legal_hits > 0:
            if prop_hits >= legal_hits:
                intent = Intent.PROPERTY_SEARCH
                confidence = min(0.6 + prop_hits * 0.08, 0.98)
            else:
                intent = Intent.LEGAL_FAQ
                confidence = min(0.6 + legal_hits * 0.08, 0.98)

            log.info(
                f"Intent (keyword): {intent.value}  "
                f"[property={prop_hits}, legal={legal_hits}, city={city_hint}]"
            )
            return IntentResult(intent, confidence, city_hint, method="keyword")

        # ── Tier 2: LLM fallback for ambiguous queries ───────────────
        return self._llm_classify(query, city_hint)

    def _detect_city(self, q_lower: str) -> Optional[str]:
        abuja_signals  = ["abuja", "fct", "maitama", "asokoro", "wuse", "gwarinpa",
                          "kubwa", "lugbe", "jabi", "garki", "utako", "kado",
                          "gwagwalada", "nyanya", "lokogoma", "durumi", "lifecamp"]
        kaduna_signals = ["kaduna", "malali", "barnawa", "sabon tasha", "rigasa",
                          "ungwan", "narayi", "kabala", "kawo", "kakuri", "sabo",
                          "tudun wada", "trikania", "television estate"]

        if any(s in q_lower for s in abuja_signals):
            return "Abuja"
        if any(s in q_lower for s in kaduna_signals):
            return "Kaduna"
        return None

    def _llm_classify(self, query: str, city_hint: Optional[str]) -> IntentResult:
        """Ask Gemini to classify — only reached for genuinely ambiguous queries."""
        log.info("Intent: ambiguous — calling LLM classifier fallback")

        prompt = f"""You are an intent classifier for PropaBridge, a Nigerian real estate platform.

Classify the user's query into EXACTLY ONE of these three intents:

PROPERTY_SEARCH  — Questions about renting/buying property, neighborhoods, prices,
                   locations, districts, apartment types (BQ, self-con, duplex, etc.)

LEGAL_FAQ        — Questions about legal documents (C of O, Deed of Assignment,
                   Tenancy Agreement, Stamp Duty, Gazette), tenant/landlord rights,
                   due diligence, mortgages, REITs, land titles.

GENERAL_CHAT     — Greetings, off-topic, unclear, or questions not related to
                   Nigerian real estate.

User query: "{query}"

Respond with a single JSON object only — no explanation, no markdown:
{{"intent": "PROPERTY_SEARCH"|"LEGAL_FAQ"|"GENERAL_CHAT", "confidence": 0.0-1.0}}"""

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=64,
                )
            )
            raw = response.text.strip()
            # Strip any accidental markdown fences
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            intent = Intent(parsed["intent"])
            confidence = float(parsed.get("confidence", 0.75))
            log.info(f"Intent (LLM): {intent.value}  confidence={confidence:.2f}")
            return IntentResult(intent, confidence, city_hint, method="llm_fallback")
        except Exception as e:
            log.warning(f"LLM classifier failed ({e}), defaulting to GENERAL_CHAT")
            return IntentResult(Intent.GENERAL_CHAT, 0.5, city_hint, method="keyword")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TOOL DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Routes the intent to the correct retrieval strategy.
    Calls query_helper.retrieve_chunks() with tuned parameters per intent.
    GENERAL_CHAT bypasses retrieval entirely.
    """

    def dispatch(self, query: str, intent_result: IntentResult) -> list[dict]:
        intent = intent_result.intent

        if intent == Intent.GENERAL_CHAT:
            log.info("Dispatch: GENERAL_CHAT — skipping vector retrieval")
            return []

        if intent == Intent.PROPERTY_SEARCH:
            log.info(
                f"Dispatch: PROPERTY_SEARCH → retrieve_chunks("
                f"top_k={Config.TOP_K_PROPERTY}, city={intent_result.city_hint})"
            )
            return retrieve_chunks(
                query_text     = query,
                top_k          = Config.TOP_K_PROPERTY,
                city_filter    = intent_result.city_hint,       # narrows to Abuja or Kaduna if detected
                category_filter= None,                          # include both neighborhood + tier_summary
            )

        if intent == Intent.LEGAL_FAQ:
            log.info(
                f"Dispatch: LEGAL_FAQ → retrieve_chunks("
                f"top_k={Config.TOP_K_LEGAL}, category=terminology)"
            )
            return retrieve_chunks(
                query_text     = query,
                top_k          = Config.TOP_K_LEGAL,
                city_filter    = None,                          # legal terms are city-agnostic
                category_filter= "terminology",                 # pin to terminology chunks only
            )

        # Unreachable but safe fallback
        return retrieve_chunks(query_text=query, top_k=Config.TOP_K_DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PROMPT AUGMENTER
# ─────────────────────────────────────────────────────────────────────────────

# PropaBridge system persona — injected into every request
SYSTEM_PROMPT = """You are PropaBridge AI, a knowledgeable and trustworthy Nigerian real estate assistant.

Your expertise covers:
- Rental and property markets in Abuja (FCT) and Kaduna
- Nigerian real estate legal documents (C of O, Deed of Assignment, Tenancy Agreement, Stamp Duty, etc.)
- Local terminology (BQ, Self-con, Face-me-I-face-you, Omo Onile, Caution Fee, etc.)
- Property due diligence best practices in Nigeria

Tone guidelines:
- Be direct, clear, and confident. No waffle.
- Always cite rent ranges in Nigerian Naira (₦).
- Flag legal risks and due diligence requirements proactively.
- Recommend consulting a licensed Estate Surveyor or lawyer for binding decisions.
- If context does not contain enough information, say so honestly — do not fabricate.
- Keep responses concise but complete. Use bullet points only when listing multiple items."""


def build_augmented_prompt(
    query: str,
    chunks: list[dict],
    intent: Intent,
) -> str:
    """
    RAG Prompt Augmentation — Retrieve → Augment pattern.
    Structures: [SYSTEM] + [RETRIEVED CONTEXT] + [USER QUESTION] + [INSTRUCTION]
    """
    parts = [SYSTEM_PROMPT, ""]

    if chunks:
        context_block = format_context_for_gemini(chunks)
        parts += [
            "━" * 60,
            "RETRIEVED KNOWLEDGE (from PropaBridge vector index):",
            "━" * 60,
            context_block,
            "━" * 60,
            "",
        ]
    else:
        # GENERAL_CHAT — no context injected
        parts += [
            "(No knowledge base context retrieved for this query.)",
            "",
        ]

    # Intent-aware instruction suffix sharpens the response
    instruction_map = {
        Intent.PROPERTY_SEARCH: (
            "Using ONLY the retrieved context above, answer the user's property question. "
            "Include specific rent ranges in ₦, tier classification, and key features of the area. "
            "If multiple neighborhoods are relevant, compare them briefly."
        ),
        Intent.LEGAL_FAQ: (
            "Using ONLY the retrieved context above, explain the legal concept clearly. "
            "Highlight any Nigerian-specific risks, required documents, and due diligence steps. "
            "Always recommend professional legal or surveying advice for final decisions."
        ),
        Intent.GENERAL_CHAT: (
            "Answer conversationally as PropaBridge AI. "
            "You may use your general knowledge about Nigerian real estate if helpful, "
            "but stay focused on the PropaBridge platform's scope (Abuja & Kaduna markets)."
        ),
    }

    parts += [
        f"USER QUESTION: {query}",
        "",
        f"INSTRUCTION: {instruction_map[intent]}",
    ]

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — GEMINI RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class GeminiGenerator:
    """Wraps the Vertex AI Gemini 1.5 Pro call with retry logic."""

    def __init__(self, model: GenerativeModel):
        self._model = model
        self._gen_config = GenerationConfig(
            temperature      = Config.TEMPERATURE,
            max_output_tokens= Config.MAX_OUTPUT_TOKENS,
            top_p            = Config.TOP_P,
        )
        # Balanced safety settings — allow real estate / legal discussion
        self._safety_settings = [
            SafetySetting(
                category  = HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold = HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            SafetySetting(
                category  = HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold = HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]

    def generate(self, augmented_prompt: str, retries: int = 2) -> str:
        last_error = None
        for attempt in range(1, retries + 2):
            try:
                log.info(f"Gemini generate — attempt {attempt}")
                response = self._model.generate_content(
                    augmented_prompt,
                    generation_config  = self._gen_config,
                    safety_settings    = self._safety_settings,
                )
                return response.text
            except Exception as e:
                last_error = e
                log.warning(f"Gemini attempt {attempt} failed: {e}")
                if attempt <= retries:
                    time.sleep(1.5 * attempt)   # exponential back-off

        raise RuntimeError(
            f"Gemini generation failed after {retries + 1} attempts: {last_error}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT ORCHESTRATOR — wires all 4 steps together
# ─────────────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    PropaBridge AI Agent — main entry point.

    Usage:
        agent = AgentOrchestrator()
        response = agent.run("How much is a 2-bedroom flat in Maitama?")
        print(response)
    """

    def __init__(self):
        log.info(f"Initialising PropaBridge Agent — project={Config.PROJECT_ID}")

        # ── Service account auth ─────────────────────────────────────
        sa_path = Config.SERVICE_ACCOUNT_FILE
        if not sa_path.exists():
            raise FileNotFoundError(
                f"service_account.json not found at: {sa_path}\n"
                "Place your GCP service account key in the project root."
            )

        credentials = service_account.Credentials.from_service_account_file(
            str(sa_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        log.info(f"Loaded service account: {credentials.service_account_email}")

        # ── Initialise Vertex AI SDK ─────────────────────────────────
        vertexai.init(
            project     = Config.PROJECT_ID,
            location    = Config.REGION,
            credentials = credentials,
        )
        log.info(f"Vertex AI initialised — model={Config.GEMINI_MODEL}")

        # ── Instantiate model once and share across components ───────
        model = GenerativeModel(Config.GEMINI_MODEL)

        # ── Wire up the 4-stage pipeline ─────────────────────────────
        self._intent_router   = IntentRouter(model)
        self._tool_dispatcher = ToolDispatcher()
        self._generator       = GeminiGenerator(model)

        log.info("Agent ready ✓")

    # ─── Public API ──────────────────────────────────────────────────

    def run(self, user_query: str) -> AgentResponse:
        """
        Execute the full RAG pipeline for a user query.
        Returns a structured AgentResponse.
        """
        if not user_query or not user_query.strip():
            raise ValueError("user_query must be a non-empty string.")

        t_start = time.monotonic()
        log.info(f"Query received: '{user_query}'")

        # ── STEP 1: Intent Routing ───────────────────────────────────
        intent_result = self._intent_router.route(user_query)

        # ── STEP 2: Tool Dispatch (Retrieve) ────────────────────────
        chunks = self._tool_dispatcher.dispatch(user_query, intent_result)
        source_keys = [c.get("source_key", "unknown") for c in chunks]

        # ── STEP 3: Prompt Augmentation (Augment) ───────────────────
        augmented_prompt = build_augmented_prompt(
            query  = user_query,
            chunks = chunks,
            intent = intent_result.intent,
        )
        log.debug(f"Augmented prompt length: {len(augmented_prompt)} chars")

        # ── STEP 4: Response Generation (Generate) ───────────────────
        answer = self._generator.generate(augmented_prompt)

        latency_ms = (time.monotonic() - t_start) * 1000
        log.info(
            f"Pipeline complete — intent={intent_result.intent.value}  "
            f"chunks={len(chunks)}  latency={latency_ms:.0f}ms"
        )

        return AgentResponse(
            answer            = answer,
            intent            = intent_result.intent,
            sources_used      = len(chunks),
            source_keys       = source_keys,
            city_filter       = intent_result.city_hint,
            latency_ms        = latency_ms,
            retrieval_skipped = (intent_result.intent == Intent.GENERAL_CHAT),
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE TEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _run_demo():
    """
    Quick smoke-test — runs representative queries across all 3 intents.
    Safe to remove once integrated into your Cloud Run service.
    """
    agent = AgentOrchestrator()

    test_queries = [
        # PROPERTY_SEARCH
        "How much does a 2-bedroom apartment cost in Maitama Abuja?",
        "What are the affordable areas to rent in Kaduna?",
        "Compare Gwarinpa and Kubwa for a family looking for a 3-bedroom flat.",
        # LEGAL_FAQ
        "What is a Certificate of Occupancy and why is it important?",
        "Explain stamp duty on a tenancy agreement in Nigeria.",
        "What is Omo Onile and how do I protect myself?",
        # GENERAL_CHAT
        "Hi, what can PropaBridge help me with?",
        "Are you better than other Nigerian property sites?",
    ]

    for query in test_queries:
        print(f"\n{'─'*62}")
        print(f"  USER: {query}")
        print(f"{'─'*62}")
        try:
            response = agent.run(query)
            print(response)
        except Exception as e:
            log.error(f"Query failed: {e}")
        time.sleep(0.5)   # gentle throttle


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single query from command line:
        # python3 agent_orchestrator.py "How much is rent in Asokoro?"
        query = " ".join(sys.argv[1:])
        agent = AgentOrchestrator()
        resp  = agent.run(query)
        print(resp)
    else:
        # Run the full demo suite
        _run_demo()
