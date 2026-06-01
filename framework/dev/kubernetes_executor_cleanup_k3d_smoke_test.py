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
"""Tests for the optional Kubernetes executor cleanup k3d smoke harness."""

from __future__ import annotations

from unittest.mock import Mock

import kubernetes_executor_cleanup_k3d_smoke as cleanup_smoke
import pytest


class _NotFound(Exception):
    status = 404


class _FakeApi:
    def __init__(self, *, pods: set[str], secrets: set[str]) -> None:
        self.pods = pods
        self.secrets = secrets

    def read_namespaced_pod(self, name: str, namespace: str) -> object:
        if name not in self.pods:
            raise _NotFound()
        return {"metadata": {"name": name, "namespace": namespace}}

    def read_namespaced_secret(self, name: str, namespace: str) -> object:
        if name not in self.secrets:
            raise _NotFound()
        return {"metadata": {"name": name, "namespace": namespace}}


def test_parse_args_uses_cleanup_environment_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cleanup smoke config defaults can come from environment variables."""
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_CLUSTER_NAME", "cluster")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_NAMESPACE", "namespace")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_CLEANUP_TIMEOUT", "3.5")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_KEEP_RESOURCES", "true")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_CLEANUP_SMOKE_DELETE_CLUSTER", "yes")

    config = cleanup_smoke.parse_args([])

    assert config.cluster_name == "cluster"
    assert config.namespace == "namespace"
    assert config.cleanup_timeout == 3.5
    assert config.keep_resources is True
    assert config.delete_cluster is True


def test_parse_args_falls_back_to_launch_smoke_cluster_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cleanup smoke can share base smoke cluster environment defaults."""
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_CLUSTER_NAME", "shared-cluster")
    monkeypatch.setenv("FLWR_K8S_EXECUTOR_SMOKE_NAMESPACE", "shared-namespace")

    config = cleanup_smoke.parse_args([])

    assert config.cluster_name == "shared-cluster"
    assert config.namespace == "shared-namespace"


def test_validate_smoke_config_rejects_invalid_timeout() -> None:
    """Test harness config rejects a non-positive cleanup timeout."""
    config = cleanup_smoke.CleanupSmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        cleanup_timeout=0,
        keep_resources=False,
        delete_cluster=False,
    )

    with pytest.raises(cleanup_smoke.SmokeFailure, match="Cleanup timeout"):
        cleanup_smoke.validate_smoke_config(config)


def test_validate_smoke_config_rejects_keep_resources_with_delete_cluster() -> None:
    """Test mutually exclusive cleanup flags are rejected."""
    config = cleanup_smoke.CleanupSmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        cleanup_timeout=1.0,
        keep_resources=True,
        delete_cluster=True,
    )

    with pytest.raises(cleanup_smoke.SmokeFailure, match="cannot be combined"):
        cleanup_smoke.validate_smoke_config(config)


def test_build_executor_config_scopes_cleanup_selector_to_run_label() -> None:
    """Test cleanup config includes local-only pool and unique run label."""
    config = cleanup_smoke.CleanupSmokeConfig(
        cluster_name="cluster",
        namespace="namespace",
        cleanup_timeout=1.0,
        keep_resources=False,
        delete_cluster=False,
    )

    executor_config = cleanup_smoke.build_executor_config(config, "run-123")

    assert executor_config.namespace == "namespace"
    assert executor_config.image == "cleanup-smoke-unused"
    assert executor_config.labels == {cleanup_smoke.RUN_LABEL_KEY: "run-123"}
    assert executor_config.resource_pool == cleanup_smoke.LOCAL_RESOURCE_POOL


def test_create_taskexecutor_pod_patches_terminal_status() -> None:
    """Test terminal synthetic Pods are patched through the status subresource."""
    api = Mock()

    cleanup_smoke.create_taskexecutor_pod(
        api,
        "namespace",
        "pod-name",
        task_id="1001",
        run_id="run-123",
        phase="Succeeded",
    )

    pod_body = api.create_namespaced_pod.call_args.kwargs["body"]
    assert pod_body["metadata"]["labels"]["flower.ai/superexec-task-id"] == "1001"
    assert pod_body["metadata"]["labels"][cleanup_smoke.RUN_LABEL_KEY] == "run-123"
    assert pod_body["spec"]["nodeSelector"] == {
        "flower.ai/nonexistent-node": "cleanup-smoke"
    }
    api.patch_namespaced_pod_status.assert_called_once()


def test_create_taskexecutor_pod_leaves_active_pod_unpatched() -> None:
    """Test active synthetic Pods are left non-terminal."""
    api = Mock()

    cleanup_smoke.create_taskexecutor_pod(
        api,
        "namespace",
        "pod-name",
        task_id="1001",
        run_id="run-123",
        phase=None,
    )

    api.patch_namespaced_pod_status.assert_not_called()


def test_create_unrelated_secret_omits_task_ownership_label() -> None:
    """Test the unrelated Secret still matches the pool but has no task id."""
    api = Mock()

    cleanup_smoke.create_unrelated_secret(api, "namespace", "secret-name", "run-123")

    secret_body = api.create_namespaced_secret.call_args.kwargs["body"]
    labels = secret_body["metadata"]["labels"]
    assert labels[cleanup_smoke.RUN_LABEL_KEY] == "run-123"
    assert labels["app.kubernetes.io/component"] == "taskexecutor"
    assert "flower.ai/superexec-task-id" not in labels


def test_prove_cleanup_result_accepts_expected_object_state() -> None:
    """Test cleanup proof passes when only expected objects remain."""
    objects = cleanup_smoke.CleanupSmokeObjects(
        terminal_pod_names=("terminal-pod",),
        terminal_secret_names=("terminal-secret",),
        active_pod_name="active-pod",
        active_secret_name="active-secret",
        orphan_secret_name="orphan-secret",
        unrelated_secret_name="unrelated-secret",
    )
    api = _FakeApi(
        pods={"active-pod"},
        secrets={"active-secret", "unrelated-secret"},
    )

    cleanup_smoke.prove_cleanup_result(
        api=api,
        namespace="namespace",
        objects=objects,
        timeout=1.0,
        poll_interval=0.01,
    )


def test_prove_cleanup_result_rejects_remaining_deleted_object() -> None:
    """Test cleanup proof fails when a terminal object remains."""
    objects = cleanup_smoke.CleanupSmokeObjects(
        terminal_pod_names=("terminal-pod",),
        terminal_secret_names=("terminal-secret",),
        active_pod_name="active-pod",
        active_secret_name="active-secret",
        orphan_secret_name="orphan-secret",
        unrelated_secret_name="unrelated-secret",
    )
    api = _FakeApi(
        pods={"terminal-pod", "active-pod"},
        secrets={"active-secret", "unrelated-secret"},
    )

    with pytest.raises(cleanup_smoke.SmokeFailure, match="expected object state"):
        cleanup_smoke.prove_cleanup_result(
            api=api,
            namespace="namespace",
            objects=objects,
            timeout=0.01,
            poll_interval=0.01,
        )
