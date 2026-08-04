"""Streamlit entry point for the v2 RAG trace demo."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rag_v2.config import ConfigError, load_config
from rag_v2.runtime import build_pipeline
from rag_v2.schemas import PipelineConfig, RagResponse


def build_view_model(response: RagResponse) -> dict[str, Any]:
    return {
        "rewritten_query": response.rewrite.rewritten_query,
        "rewrite_fallback": response.rewrite.used_fallback,
        "retrieved": [
            {
                "chunk_id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "section": item.chunk.section,
                "source": item.chunk.source,
                "text": item.chunk.text,
                "score": item.score,
                "retriever": item.retriever,
                "metadata": item.metadata,
            }
            for item in response.retrieved
        ],
        "answer": response.generation.answer,
        "citations": response.generation.cited_chunk_ids,
        "abstained": response.generation.abstained,
        "trace": response.trace,
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RAG v2 Trace Demo", layout="wide")
    st.title("RAG v2 检索链路演示")
    st.caption("展示 Query Rewrite、召回、Rerank、证据引用和阶段耗时")

    config_path = Path(os.getenv("V2_CONFIG_PATH", "config/v2.yaml"))
    try:
        config = load_config(config_path)
        config.validate(requires_generation=True)
    except ConfigError as exc:
        st.error(str(exc))
        st.info("请复制 .env.example 为 .env，填写 DEEPSEEK_API_KEY，并先构建索引。")
        return

    mode = st.selectbox("检索方案", ["hybrid", "bm25", "dense"], index=0)
    question = st.text_input("输入问题", placeholder="例如：智能科技专业主要学什么？")
    if not question:
        st.info("输入问题后开始检索。")
        return

    pipeline_config = PipelineConfig(
        corpus_dir=config.corpus_dir,
        artifact_dir=config.artifact_dir,
        embedding_model=config.embedding_model,
        reranker_model=config.reranker_model,
        mode=mode,
        top_k=config.top_k,
        rerank_top_k=config.rerank_top_k,
    )
    try:
        pipeline = build_pipeline(config)
        response = pipeline.run(question, [], pipeline_config)
    except Exception as exc:
        st.error(f"运行失败：{exc}")
        return

    view = build_view_model(response)
    st.subheader("回答")
    st.write(view["answer"])
    st.caption(f"引用：{', '.join(view['citations']) or '无'}")

    left, right = st.columns(2)
    with left:
        st.subheader("查询改写")
        st.code(view["rewritten_query"])
        st.subheader("召回与重排结果")
        for item in view["retrieved"]:
            with st.expander(f"{item['chunk_id']} · {item['retriever']} · {item['score']:.4f}"):
                st.write(item["text"])
                st.caption(f"来源：{item['source']} / {item['section']}")
    with right:
        st.subheader("阶段 Trace")
        st.json(view["trace"])


if __name__ == "__main__":
    main()
