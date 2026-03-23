"""
Knowledge graph ingestion pipeline for test_runs.

Pipeline steps:
    1. schema_selector  — select relevant schema chunks for the report.
    2. extractor        — extract structured entities with an LLM agent.
    3. searchtext       — generate searchText per node (LLM or field-concat).
    4. embedder         — batch-embed all searchTexts.
    5. neo4j_writer     — MERGE nodes and relationships into Neo4j.

Entry points:
    run_kg_ingestion.run_kg_ingestion   — called from run_mission.py or standalone.
    ingest_report.main                  — standalone CLI.
    setup_indexes.main                  — one-time vector index creation.
"""
