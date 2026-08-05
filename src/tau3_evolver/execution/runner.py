from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

from tau3_evolver.artifacts.jsonl import JsonlWriter, iter_jsonl_objects
from tau3_evolver.artifacts.run import episode_artifact_metadata, write_run_record
from tau3_evolver.benchmarks import PreparedBenchmark, benchmark_registry
from tau3_evolver.config import ProjectConfig, load_config
from tau3_evolver.execution.batch import run_batch
from tau3_evolver.execution.request import ExecutionRequest
from tau3_evolver.execution.results import BatchResult
from tau3_evolver.evaluation.metrics import compute_reward_metrics
from tau3_evolver.agent.policy import FastLoopConfig
from tau3_evolver.memory.embeddings import build_embedding_provider
from tau3_evolver.memory.retrieval import Retriever
from tau3_evolver.memory.snapshots import resolve_memory
from tau3_evolver.models.openai_compatible import (
    OpenAICompatibleFastLoopPolicy,
    OpenAICompatibleHttpClient,
)


def execute(request: ExecutionRequest) -> BatchResult:
    config = load_config(request.config_path, request.overrides)
    prepared = benchmark_registry.resolve(request.benchmark.value).prepare(
        config, request.mode
    )
    prepared = _select_execution_tasks(prepared, request=request, config=config)
    run_dir = (request.output_root / request.run_id).resolve()
    if run_dir.exists():
        raise FileExistsError(f"run output already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    memory = resolve_memory(request, prepared)
    if memory.source is not None:
        embedding = build_embedding_provider(config.memory)
        retriever = Retriever(embedding)
    else:
        retriever = None

    policy = _policy(config)
    episodes_path = run_dir / "episodes.jsonl"
    result = run_batch(
        prepared=prepared,
        request=request,
        project_config=config,
        policy=policy,
        repository=memory.source,
        destination_repository=memory.destination,
        retriever=retriever,
        fast_loop_config=_fast_loop_config(config, request.memory_enabled),
        input_memory_snapshot_id=memory.input_snapshot_id,
        memory_generation=memory.generation,
        episode_writer=JsonlWriter(episodes_path),
    )
    artifact = episode_artifact_metadata(episodes_path)
    if artifact["rows"] != len(prepared.task_ids):
        raise RuntimeError("episode artifact does not cover the prepared task set")
    write_run_record(
        run_dir / "run.json",
        {
            "run_id": request.run_id,
            "status": "failed" if result.failures else "completed",
            "execution": {
                "benchmark": prepared.name,
                "mode": request.mode.value,
                "split": prepared.split_name,
                "split_hash": prepared.split_hash,
                "task_scope": "debug" if request.debug else "full",
                "planned_task_count": len(prepared.task_ids),
            },
            "runtime": {
                "source_root": str(prepared.runtime_origin.source_root),
                "package_version": prepared.runtime_origin.package_version,
                "git_commit": prepared.runtime_origin.git_commit,
            },
            "policy": {
                "model_revision": config.model.served_model_name,
                "checkpoint": (
                    str(request.checkpoint)
                    if request.checkpoint is not None
                    else None
                ),
            },
            "memory": {
                "enabled": request.memory_enabled,
                "generation": memory.generation,
                "source_namespace": memory.source_namespace,
                "destination_namespace": memory.destination_namespace,
                "input_snapshot_id": result.input_memory_snapshot_id,
                "output_snapshot_id": result.output_memory_snapshot_id,
                "cross_domain": request.is_cross_domain_memory(
                    prepared.default_memory_namespace
                ),
            },
            "config": config.model_dump(mode="json"),
            "summary": {
                "metrics": asdict(compute_reward_metrics(result)),
                **_memory_summary(episodes_path),
            },
            "artifacts": {"episodes": artifact},
        },
    )
    return result


def _select_execution_tasks(
    prepared: PreparedBenchmark,
    *,
    request: ExecutionRequest,
    config: ProjectConfig,
) -> PreparedBenchmark:
    if request.debug:
        if config.execution.max_concurrency < 2:
            raise ValueError(
                "debug runs require execution.max_concurrency to be at least 2"
            )
        if len(prepared.task_ids) < 2:
            raise ValueError("debug runs require at least two benchmark tasks")
        return prepared.first_tasks(config.execution.max_concurrency)
    return prepared


def _policy(config: ProjectConfig) -> OpenAICompatibleFastLoopPolicy:
    api_key = os.environ.get(config.model.api_key_env) or "EMPTY"
    client = OpenAICompatibleHttpClient(
        base_url=config.model.serving_base_url,
        model=config.model.served_model_name,
        api_key=api_key,
        max_tokens=config.model.max_tokens,
        generation_settings=config.model.generation_settings,
        request_timeout_s=config.model.request_timeout_s,
    )
    return OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )


def _fast_loop_config(config: ProjectConfig, memory_enabled: bool) -> FastLoopConfig:
    memory = config.memory
    return FastLoopConfig(
        retrieve_top_k=memory.retrieve_top_k,
        max_episode_steps=config.rollout.max_episode_steps,
        memory_enabled=memory_enabled,
        max_new_tips_per_episode=memory.max_new_tips_per_episode,
        max_new_skills_per_episode=memory.max_new_skills_per_episode,
        max_new_tools_per_episode=memory.max_new_tools_per_episode,
        max_new_trajectories_per_episode=memory.max_new_trajectories_per_episode,
        maintenance_tip_capacity=memory.maintenance_tip_capacity,
        maintenance_similarity_threshold=memory.maintenance_similarity_threshold,
        maintenance_priority_pair_limit=memory.maintenance_priority_pair_limit,
        retrieval_mmr_lambda_tip=memory.retrieval_mmr_lambda_tip,
        retrieval_mmr_lambda_skill=memory.retrieval_mmr_lambda_skill,
        retrieval_mmr_lambda_tool=memory.retrieval_mmr_lambda_tool,
        retrieval_mmr_lambda_trajectory=memory.retrieval_mmr_lambda_trajectory,
        retrieval_global_mmr_lambda=memory.retrieval_global_mmr_lambda,
        retrieval_quota_tip=memory.retrieval_quota_tip,
        retrieval_quota_skill=memory.retrieval_quota_skill,
        retrieval_quota_tool=memory.retrieval_quota_tool,
        retrieval_quota_trajectory=memory.retrieval_quota_trajectory,
        selection_max_total=memory.selection_max_total,
        selection_max_tip=memory.selection_max_tip,
        selection_max_skill=memory.selection_max_skill,
        selection_max_tool=memory.selection_max_tool,
        selection_max_trajectory=memory.selection_max_trajectory,
    )


def _memory_summary(path: Path) -> dict[str, Any]:
    retrieved: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    incompatible_tool_memory_count = 0
    for episode in iter_jsonl_objects(path):
        if episode.get("status") != "completed":
            continue
        memory = episode.get("memory")
        if not isinstance(memory, dict) or memory.get("enabled") is not True:
            continue
        retrieval = memory.get("retrieval")
        if isinstance(retrieval, dict):
            candidates = retrieval.get("candidates", [])
            candidate_tiers = {
                str(item["memory_id"]): str(item["tier"])
                for item in candidates
                if isinstance(item, dict) and "memory_id" in item and "tier" in item
            }
            retrieved.update(candidate_tiers.values())
            incompatible_tool_memory_count += int(
                retrieval.get("incompatible_tool_memory_count", 0)
            )
            selected.update(
                candidate_tiers[memory_id]
                for memory_id in memory.get("selected_memory_ids", [])
                if memory_id in candidate_tiers
            )
    return {
        "retrieved_counts_by_tier": dict(sorted(retrieved.items())),
        "selected_counts_by_tier": dict(sorted(selected.items())),
        "incompatible_tool_memory_count": incompatible_tool_memory_count,
    }
