# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Optional local k3d smoke harness for Kubernetes TaskExecutor cleanup."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flwr.supercore.superexec.executor.kubernetes_executor import (
    CompletedPodSweeper,
    KubernetesExecutorConfig,
    _capacity_label_selector,
)

from kubernetes_executor_k3d_smoke import (
    CoreV1ApiAdapter,
    ENV_PREFIX as BASE_ENV_PREFIX,
    SkipSmoke,
    SmokeFailure,
    _check_local_tools,
    _load_kubernetes,
    _raise_unless_not_found,
    _run_command,
    cleanup_labeled_objects,
    ensure_k3d_cluster,
    ensure_namespace,
)

DEFAULT_CLUSTER_NAME = "flwr-k8s-executor-smoke"
DEFAULT_NAMESPACE = "flwr-k8s-executor-smoke"
DEFAULT_CLEANUP_TIMEOUT = 30.0
LOCAL_RESOURCE_POOL = "local-k3d-cleanup-smoke"
RUN_LABEL_KEY = "flower.ai/k8s-executor-smoke-run"
ENV_PREFIX = "FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_"


@dataclass(frozen=True)
class CleanupSmokeConfig:
    """Configuration for one optional local cleanup smoke run."""

    cluster_name: str
    namespace: str
    cleanup_timeout: float
    keep_resources: bool
    delete_cluster: bool


@dataclass(frozen=True)
class CleanupSmokeObjects:
    """Names created for one cleanup smoke proof."""

    terminal_pod_names: tuple[str, ...]
    terminal_secret_names: tuple[str, ...]
    active_pod_name: str
    active_secret_name: str
    orphan_secret_name: str
    unrelated_secret_name: str


def parse_args(argv: Sequence[str] | None = None) -> CleanupSmokeConfig:
    """Parse CLI args and environment defaults for the cleanup smoke harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the optional local k3d smoke harness for "
            "CompletedPodSweeper cleanup behavior."
        )
    )
    parser.add_argument(
        "--cluster-name",
        default=_env("CLUSTER_NAME", DEFAULT_CLUSTER_NAME),
        help="Local k3d cluster name.",
    )
    parser.add_argument(
        "--namespace",
        default=_env("NAMESPACE", DEFAULT_NAMESPACE),
        help="Namespace used for cleanup smoke objects.",
    )
    parser.add_argument(
        "--cleanup-timeout",
        type=float,
        default=_env_float("CLEANUP_TIMEOUT", DEFAULT_CLEANUP_TIMEOUT),
        help="Harness-level timeout for observing cleanup results.",
    )
    parser.add_argument(
        "--keep-resources",
        action="store_true",
        default=_env_bool("KEEP_RESOURCES", False),
        help="Keep remaining smoke objects for debugging.",
    )
    parser.add_argument(
        "--delete-cluster",
        action="store_true",
        default=_env_bool("DELETE_CLUSTER", False),
        help="Delete the k3d cluster at the end only if this harness created it.",
    )
    args = parser.parse_args(argv)

    return CleanupSmokeConfig(
        cluster_name=args.cluster_name,
        namespace=args.namespace,
        cleanup_timeout=args.cleanup_timeout,
        keep_resources=args.keep_resources,
        delete_cluster=args.delete_cluster,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional local cleanup smoke harness."""
    config = parse_args(argv)
    try:
        run_smoke(config)
    except SkipSmoke as exc:
        print(f"SKIP: {exc}")
        return 0
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


def run_smoke(config: CleanupSmokeConfig) -> None:
    """Run one local Kubernetes cleanup smoke proof."""
    validate_smoke_config(config)
    _check_local_tools()
    client_module, kube_config = _load_kubernetes()

    api: Any | None = None
    selector: str | None = None
    cluster_created = ensure_k3d_cluster(config.cluster_name)
    try:
        kube_config.load_kube_config()
        api = client_module.CoreV1Api()
        ensure_namespace(api, config.namespace)

        run_id = uuid.uuid4().hex
        executor_config = build_executor_config(config, run_id)
        selector = _capacity_label_selector(executor_config)
        objects = create_cleanup_smoke_objects(api, config.namespace, run_id)

        print(f"Using cluster: {config.cluster_name}")
        print(f"Using namespace: {config.namespace}")
        print(f"Using cleanup selector: {selector}")

        CompletedPodSweeper(
            client=CoreV1ApiAdapter(api), config=executor_config
        ).sweep()
        prove_cleanup_result(
            api=api,
            namespace=config.namespace,
            objects=objects,
            timeout=config.cleanup_timeout,
        )

        print("Kubernetes executor cleanup k3d smoke harness passed.")
        print(
            "Cleanup check: "
            f"kubectl get pods,secrets -n {config.namespace} -l '{selector}'"
        )
    finally:
        if not config.keep_resources:
            try:
                if api is not None and selector is not None:
                    cleanup_labeled_objects(api, config.namespace, selector)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"WARN: cleanup failed: {exc}", file=sys.stderr)
        if config.delete_cluster and cluster_created:
            _run_command(["k3d", "cluster", "delete", config.cluster_name])
        elif config.delete_cluster and not cluster_created:
            print(
                "Preserving reused k3d cluster despite --delete-cluster: "
                f"{config.cluster_name}"
            )


def validate_smoke_config(config: CleanupSmokeConfig) -> None:
    """Validate harness-local cleanup smoke configuration."""
    for name, value in (
        ("Cluster name", config.cluster_name),
        ("Namespace", config.namespace),
    ):
        if not value.strip():
            raise SmokeFailure(f"{name} must not be empty.")
    if config.cleanup_timeout <= 0:
        raise SmokeFailure("Cleanup timeout must be positive.")
    if config.keep_resources and config.delete_cluster:
        raise SmokeFailure("--keep-resources cannot be combined with --delete-cluster.")


def build_executor_config(
    config: CleanupSmokeConfig, run_id: str
) -> KubernetesExecutorConfig:
    """Build the executor config used for one cleanup smoke run."""
    return KubernetesExecutorConfig(
        namespace=config.namespace,
        image="cleanup-smoke-unused",
        resource_pool=LOCAL_RESOURCE_POOL,
        labels={RUN_LABEL_KEY: run_id},
        completed_pod_retention_seconds=0.0,
    )


def create_cleanup_smoke_objects(
    api: Any, namespace: str, run_id: str
) -> CleanupSmokeObjects:
    """Create synthetic TaskExecutor Pods and Secrets for cleanup proof."""
    suffix = run_id[:12]
    succeeded_pod = f"flwr-cleanup-succeeded-{suffix}"
    failed_pod = f"flwr-cleanup-failed-{suffix}"
    active_pod = f"flwr-cleanup-active-{suffix}"
    succeeded_secret = f"{succeeded_pod}-appio"
    failed_secret = f"{failed_pod}-appio"
    active_secret = f"{active_pod}-appio"
    orphan_secret = f"flwr-cleanup-orphan-{suffix}-appio"
    unrelated_secret = f"flwr-cleanup-unrelated-{suffix}"

    create_taskexecutor_pod(
        api, namespace, succeeded_pod, task_id="1001", run_id=run_id, phase="Succeeded"
    )
    create_taskexecutor_secret(
        api, namespace, succeeded_secret, task_id="1001", run_id=run_id
    )
    create_taskexecutor_pod(
        api, namespace, failed_pod, task_id="1002", run_id=run_id, phase="Failed"
    )
    create_taskexecutor_secret(
        api, namespace, failed_secret, task_id="1002", run_id=run_id
    )
    create_taskexecutor_pod(
        api, namespace, active_pod, task_id="1003", run_id=run_id, phase=None
    )
    create_taskexecutor_secret(
        api, namespace, active_secret, task_id="1003", run_id=run_id
    )
    create_taskexecutor_secret(
        api, namespace, orphan_secret, task_id="1004", run_id=run_id
    )
    create_unrelated_secret(api, namespace, unrelated_secret, run_id)

    return CleanupSmokeObjects(
        terminal_pod_names=(succeeded_pod, failed_pod),
        terminal_secret_names=(succeeded_secret, failed_secret),
        active_pod_name=active_pod,
        active_secret_name=active_secret,
        orphan_secret_name=orphan_secret,
        unrelated_secret_name=unrelated_secret,
    )


def create_taskexecutor_pod(
    api: Any,
    namespace: str,
    name: str,
    *,
    task_id: str,
    run_id: str,
    phase: str | None,
) -> None:
    """Create one synthetic TaskExecutor Pod and optionally patch it terminal."""
    body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": _task_labels(task_id=task_id, run_id=run_id),
        },
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"flower.ai/nonexistent-node": "cleanup-smoke"},
            "containers": [
                {
                    "name": "taskexecutor",
                    "image": "example.invalid/flwr-cleanup-smoke:never",
                    "imagePullPolicy": "Never",
                    "command": ["true"],
                }
            ],
        },
    }
    api.create_namespaced_pod(namespace=namespace, body=body)
    if phase is not None:
        patch_taskexecutor_pod_status(api, namespace, name, phase)


def patch_taskexecutor_pod_status(
    api: Any, namespace: str, name: str, phase: str
) -> None:
    """Patch a synthetic Pod into a terminal phase without pulling an image."""
    api.patch_namespaced_pod_status(
        name=name,
        namespace=namespace,
        body={
            "status": {
                "phase": phase,
                "containerStatuses": [
                    {
                        "name": "taskexecutor",
                        "state": {
                            "terminated": {
                                "exitCode": 0 if phase == "Succeeded" else 1,
                                "finishedAt": "2026-05-27T19:00:00Z",
                            }
                        },
                    }
                ],
            }
        },
    )


def create_taskexecutor_secret(
    api: Any, namespace: str, name: str, *, task_id: str, run_id: str
) -> None:
    """Create one synthetic per-task AppIo Secret."""
    body = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": _task_labels(task_id=task_id, run_id=run_id),
        },
        "type": "Opaque",
        "stringData": {"token": f"cleanup-smoke-token-{task_id}"},
    }
    api.create_namespaced_secret(namespace=namespace, body=body)


def create_unrelated_secret(api: Any, namespace: str, name: str, run_id: str) -> None:
    """Create a selector-matching Secret without a per-task ownership label."""
    body = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": _pool_labels(run_id),
        },
        "type": "Opaque",
        "stringData": {"token": "do-not-delete"},
    }
    api.create_namespaced_secret(namespace=namespace, body=body)


def prove_cleanup_result(
    *,
    api: Any,
    namespace: str,
    objects: CleanupSmokeObjects,
    timeout: float,
    poll_interval: float = 0.5,
) -> None:
    """Verify cleanup deleted only the eligible smoke objects."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        deleted_names = (
            *objects.terminal_pod_names,
            *objects.terminal_secret_names,
            objects.orphan_secret_name,
        )
        kept_names = (
            objects.active_pod_name,
            objects.active_secret_name,
            objects.unrelated_secret_name,
        )
        if all(
            not object_exists(api, namespace, name) for name in deleted_names
        ) and all(object_exists(api, namespace, name) for name in kept_names):
            return
        time.sleep(poll_interval)

    raise SmokeFailure(
        "Cleanup proof did not observe the expected object state before timeout."
    )


def object_exists(api: Any, namespace: str, name: str) -> bool:
    """Return true if a Pod or Secret with the given name exists."""
    return pod_exists(api, namespace, name) or secret_exists(api, namespace, name)


def pod_exists(api: Any, namespace: str, name: str) -> bool:
    """Return true if a Pod exists."""
    try:
        api.read_namespaced_pod(name=name, namespace=namespace)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _raise_unless_not_found(exc)
        return False


def secret_exists(api: Any, namespace: str, name: str) -> bool:
    """Return true if a Secret exists."""
    try:
        api.read_namespaced_secret(name=name, namespace=namespace)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _raise_unless_not_found(exc)
        return False


def _env(name: str, default: str) -> str:
    """Return a cleanup smoke harness environment override."""
    value = os.getenv(f"{ENV_PREFIX}{name}")
    if value is not None:
        return value
    return os.getenv(f"{BASE_ENV_PREFIX}{name}", default)


def _env_float(name: str, default: float) -> float:
    """Return a float cleanup smoke harness environment override."""
    value = _env(name, "")
    if not value:
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean cleanup smoke harness environment override."""
    value = _env(name, "")
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _pool_labels(run_id: str) -> dict[str, str]:
    """Return labels that scope cleanup smoke objects to one run."""
    return {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/resource-pool": LOCAL_RESOURCE_POOL,
        RUN_LABEL_KEY: run_id,
    }


def _task_labels(*, task_id: str, run_id: str) -> dict[str, str]:
    """Return TaskExecutor ownership labels for cleanup smoke objects."""
    labels = _pool_labels(run_id)
    labels["flower.ai/superexec-task-id"] = task_id
    labels["flower.ai/task-type"] = "flwr-serverapp"
    return labels


if __name__ == "__main__":
    sys.exit(main())
