import copy
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from data_utils import SD2_instance_generator, generate_due_dates
from global_env import EventBurstGenerator, GlobalTimelineOrchestrator, JobSpec, split_matrix_to_jobs


def fixed_k_sampler(k: int):
    def _fn(rng: np.random.Generator) -> int:
        return int(k)

    return _fn


def seed_dynamic_simulation(seed: int) -> np.random.Generator:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def create_dynamic_generator(config, *, interarrival_mean: float, burst_k: int, rng: np.random.Generator) -> EventBurstGenerator:
    base_cfg = copy.deepcopy(config)
    return EventBurstGenerator(
        SD2_instance_generator,
        base_cfg,
        int(config.n_m),
        float(interarrival_mean),
        fixed_k_sampler(int(burst_k)),
        rng,
    )


def create_dynamic_world(config, *, interarrival_mean: float, burst_k: int, seed: Optional[int] = None):
    if seed is None:
        seed = int(getattr(config, "event_seed", 42))
    rng = seed_dynamic_simulation(int(seed))
    gen = create_dynamic_generator(config, interarrival_mean=float(interarrival_mean), burst_k=int(burst_k), rng=rng)
    orch = GlobalTimelineOrchestrator(int(config.n_m), gen, t0=0.0)
    return rng, gen, orch


def sample_initial_jobs(config, *, rng: np.random.Generator, base_job_id: int = 0, t_arrive: float = 0.0) -> List[JobSpec]:
    init_jobs = int(getattr(config, "init_jobs", 0))
    if init_jobs <= 0:
        return []

    init_cfg = copy.deepcopy(config)
    setattr(init_cfg, "n_j", init_jobs)
    jl, pt, _ = SD2_instance_generator(init_cfg, rng=rng)
    dd_rel = generate_due_dates(
        jl,
        pt,
        tightness=getattr(config, "due_date_tightness", 1.2),
        due_date_mode="k",
        rng=rng,
    )
    return split_matrix_to_jobs(jl, pt, base_job_id=base_job_id, t_arrive=float(t_arrive), due_dates=float(t_arrive) + dd_rel)


def register_initial_jobs(
    orch: GlobalTimelineOrchestrator,
    gen: EventBurstGenerator,
    init_jobs: List[JobSpec],
    all_job_due_dates: Dict[int, float],
    *,
    t0: float = 0.0,
) -> int:
    if not init_jobs:
        return 0

    for job in init_jobs:
        all_job_due_dates[job.job_id] = float(job.meta["due_date"])
    orch.buffer.extend(init_jobs)
    gen.bump_next_id(max((job.job_id for job in init_jobs), default=-1) + 1)
    orch.event_release_and_reschedule(float(t0))
    return 1


@dataclass
class DynamicArrivalEvent:
    event_id: int
    time: float
    inter_arrival: float
    jobs: List[JobSpec]


def generate_dynamic_job_stream(
    config,
    *,
    max_events: int,
    interarrival_mean: float,
    burst_k: int = 1,
    seed: Optional[int] = None,
):
    if seed is None:
        seed = int(getattr(config, "event_seed", 42))

    rng = seed_dynamic_simulation(int(seed))
    gen = create_dynamic_generator(config, interarrival_mean=float(interarrival_mean), burst_k=int(burst_k), rng=rng)
    all_job_due_dates: Dict[int, float] = {}
    init_jobs = sample_initial_jobs(config, rng=rng, base_job_id=0, t_arrive=0.0)
    for job in init_jobs:
        all_job_due_dates[job.job_id] = float(job.meta["due_date"])
    if init_jobs:
        gen.bump_next_id(max((job.job_id for job in init_jobs), default=-1) + 1)

    events: List[DynamicArrivalEvent] = []
    t_prev = 0.0
    t_now = 0.0
    t_next = float(gen.sample_next_time(t_now))

    for event_id in range(1, int(max_events) + 1):
        t_now = float(t_next)
        new_jobs = gen.generate_burst(t_now)
        for job in new_jobs:
            all_job_due_dates[job.job_id] = float(job.meta["due_date"])
        events.append(
            DynamicArrivalEvent(
                event_id=event_id,
                time=t_now,
                inter_arrival=float(t_now - t_prev),
                jobs=new_jobs,
            )
        )
        t_prev = t_now
        t_next = float(gen.sample_next_time(t_now))

    all_jobs = list(init_jobs)
    for event in events:
        all_jobs.extend(event.jobs)

    return {
        "seed": int(seed),
        "init_jobs": init_jobs,
        "events": events,
        "all_jobs": all_jobs,
        "all_job_due_dates": all_job_due_dates,
    }


def job_to_dict(job: JobSpec) -> Dict:
    operations = []
    for op_id, op in enumerate(job.operations):
        machine_times = {}
        if op.time_row is not None:
            for m_idx, pt in enumerate(op.time_row):
                if float(pt) > 0:
                    machine_times[str(int(m_idx))] = float(pt)
        elif op.machine_times is not None:
            for m_idx, pt in op.machine_times.items():
                if float(pt) > 0:
                    machine_times[str(int(m_idx))] = float(pt)
        operations.append({"op_id": int(op_id), "machine_times": machine_times})

    return {
        "job_id": int(job.job_id),
        "arrive_time": float(job.meta.get("t_arrive", 0.0)),
        "due_date": float(job.meta.get("due_date", 0.0)),
        "total_proc_time": float(job.meta.get("total_proc_time", 0.0)),
        "min_total_proc_time": float(job.meta.get("min_total_proc_time", 0.0)),
        "total_ops": int(job.meta.get("total_ops", len(job.operations))),
        "operations": operations,
    }


def arrival_event_to_dict(event: DynamicArrivalEvent) -> Dict:
    return {
        "event_id": int(event.event_id),
        "time": float(event.time),
        "inter_arrival": float(event.inter_arrival),
        "job_ids": [int(job.job_id) for job in event.jobs],
        "jobs": [job_to_dict(job) for job in event.jobs],
    }


def dynamic_job_stream_to_dict(stream: Dict, config=None) -> Dict:
    meta = {
        "seed": int(stream["seed"]),
        "n_machines": int(getattr(config, "n_m", 0)) if config is not None else 0,
        "init_jobs_count": len(stream["init_jobs"]),
        "event_count": len(stream["events"]),
    }
    if config is not None:
        meta.update(
            {
                "interarrival_mean": float(getattr(config, "interarrival_mean", 0.0)),
                "burst_size": int(getattr(config, "burst_size", 1)),
                "event_horizon": int(getattr(config, "event_horizon", len(stream["events"]))),
                "due_date_tightness": float(getattr(config, "due_date_tightness", 1.2)),
                "init_jobs": int(getattr(config, "init_jobs", 0)),
            }
        )

    return {
        "meta": meta,
        "init_jobs": [job_to_dict(job) for job in stream["init_jobs"]],
        "events": [arrival_event_to_dict(event) for event in stream["events"]],
        "jobs": [job_to_dict(job) for job in stream["all_jobs"]],
        "all_job_due_dates": {str(k): float(v) for k, v in stream["all_job_due_dates"].items()},
    }
