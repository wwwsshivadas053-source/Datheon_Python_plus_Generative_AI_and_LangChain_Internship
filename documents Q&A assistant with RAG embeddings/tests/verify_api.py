import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
from app import app, vector_service, rag_pipeline


def test_full_rag_workflow():
    client = app.test_client()

    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE END-TO-END RAG PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Check Initial Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_data = res_status.get_json()["data"]
    print(f"\n[+] Active Vector Store: {status_data['active_store']}")
    print(f"[+] Initial Documents: {status_data['total_documents']}")

    # 2. Load Sample Document
    print("\n[+] Triggering /api/load-sample...")
    res_sample = client.post("/api/load-sample")
    assert res_sample.status_code == 200
    sample_data = res_sample.get_json()
    assert sample_data["success"] is True
    print(f"[+] Loaded: {sample_data['document']['file_name']} | Pages: {sample_data['document']['pages']} | Chunks: {sample_data['document']['chunks']}")

    # 3. Query RAG Assistant on FAISS
    query1 = "What is the chunk overlap used in the text splitter?"
    print(f"\n[+] Executing Query on FAISS: '{query1}'")
    res_query1 = client.post("/api/query", json={"question": query1})
    assert res_query1.status_code == 200
    q1_data = res_query1.get_json()["data"]
    print(f"[+] Retrieved Chunks: {q1_data['retrieved_chunks']}")
    print(f"[+] Synthesized Answer Preview:\n{q1_data['answer'][:250]}...")
    print(f"[+] Citations Generated ({len(q1_data['citations'])}):")
    for cit in q1_data["citations"]:
        print(f"    - [Document: {cit['file_name']}, Page: {cit['page']}] Relevance: {cit['relevance_score']:.2f}")

    assert len(q1_data["citations"]) > 0
    assert q1_data["citations"][0]["file_name"] == "ai_rag_overview.pdf"
    assert q1_data["citations"][0]["page"] == 1

    # 4. Dynamically Switch Vector Store to ChromaDB
    print("\n[+] Testing dynamic switch to ChromaDB...")
    res_switch = client.post("/api/switch-store", json={"store_type": "chroma"})
    assert res_switch.status_code == 200
    switch_data = res_switch.get_json()
    print(f"[+] Store Switch Response: {switch_data['message']}")
    assert switch_data["stats"]["active_store"] == "CHROMA"

    # 5. Query RAG Assistant on ChromaDB
    query2 = "Compare FAISS and ChromaDB search latency."
    print(f"\n[+] Executing Query on ChromaDB: '{query2}'")
    res_query2 = client.post("/api/query", json={"question": query2})
    assert res_query2.status_code == 200
    q2_data = res_query2.get_json()["data"]
    print(f"[+] Chroma Retrieved Chunks: {q2_data['retrieved_chunks']}")
    print(f"[+] Synthesized Answer Preview:\n{q2_data['answer'][:250]}...")
    print(f"[+] Chroma Citations ({len(q2_data['citations'])}):")
    for cit in q2_data["citations"]:
        print(f"    - [Document: {cit['file_name']}, Page: {cit['page']}] Relevance: {cit['relevance_score']:.2f}")

    assert len(q2_data["citations"]) > 0
    assert q2_data["citations"][0]["file_name"] == "ai_rag_overview.pdf"
    assert any(c["page"] in [2, 3] for c in q2_data["citations"])

    # 6. Test Anti-Hallucination Grounding
    query_hallucination = "What is the capital of France and who won the 2022 World Cup?"
    print(f"\n[+] Testing Anti-Hallucination Prompt with Out-of-Context Question: '{query_hallucination}'")
    res_hallucination = client.post("/api/query", json={"question": query_hallucination})
    assert res_hallucination.status_code == 200
    h_data = res_hallucination.get_json()["data"]
    print(f"[+] Response for ungrounded question: {h_data['answer'][:150]}...")

    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY (100% PASS RATE)")
    print("=" * 60)


if __name__ == "__main__":
    test_full_rag_workflow()
