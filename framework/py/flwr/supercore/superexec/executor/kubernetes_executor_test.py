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
"""Tests for SuperExec Kubernetes executor."""

from typing import Any
from unittest.mock import Mock, call

import pytest

from flwr.supercore.constant import TaskType

from .kubernetes_executor import (
    APPIO_CREDENTIALS_MOUNT_PATH,
    APPIO_ROOT_CERTIFICATES_FILE_PATH,
    APPIO_TOKEN_FILE_PATH,
    KubernetesExecutor,
    KubernetesExecutorConfig,
    build_appio_credentials_secret,
    build_taskexecutor_pod,
)
from .types import ExecutionSpec, LaunchResultStatus


class _KubernetesApiError(Exception):
    """Minimal Kubernetes client error used by executor tests."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


def _execution_spec(**overrides: Any) -> ExecutionSpec:
    base: dict[str, Any] = {
        "task_type": TaskType.SERVER_APP,
        "appio_api_address": "appio.example.com:9092",
        "token": "task-token",
        "insecure": False,
        "root_certificates_path": None,
        "runtime_dependency_install": False,
        "parent_pid": None,
        "suppress_output": True,
        "task_id": 123,
    }
    base.update(overrides)
    return ExecutionSpec(**base)


def _executor_config(**overrides: Any) -> KubernetesExecutorConfig:
    base: dict[str, Any] = {
        "namespace": "flower-system",
        "image": "ghcr.io/flwrlabs/taskexecutor:dev",
        "appio_root_certificates": "root-ca",
    }
    base.update(overrides)
    return KubernetesExecutorConfig(**base)


def _pod(phase: str, deletion_timestamp: str | None = None) -> dict[str, Any]:
    metadata = {}
    if deletion_timestamp is not None:
        metadata["deletionTimestamp"] = deletion_timestamp
    return {"metadata": metadata, "status": {"phase": phase}}


def _contains_value(value: Any, needle: str) -> bool:
    """Return true if a nested Kubernetes object contains a value."""
    if value == needle:
        return True
    if isinstance(value, dict):
        return any(
            _contains_value(nested_key, needle)
            or _contains_value(nested_value, needle)
            for nested_key, nested_value in value.items()
        )
    if isinstance(value, list):
        return any(_contains_value(item, needle) for item in value)
    return False


def test_build_appio_credentials_secret_contains_token_and_ca() -> None:
    """Test building the AppIo credential Secret."""
    secret = build_appio_credentials_secret(_execution_spec(), _executor_config())

    assert secret == {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": "flwr-taskexecutor-123-appio",
            "namespace": "flower-system",
            "labels": {
                "app.kubernetes.io/name": "flower",
                "app.kubernetes.io/component": "taskexecutor",
                "flower.ai/superexec-task-id": "123",
                "flower.ai/task-type": "flwr-serverapp",
            },
        },
        "type": "Opaque",
        "stringData": {"token": "task-token", "ca.crt": "root-ca"},
    }


def test_build_taskexecutor_pod_uses_secret_files_for_credentials() -> None:
    """Test Pod construction uses mounted files instead of credential args."""
    pod = build_taskexecutor_pod(_execution_spec(), _executor_config())
    container = pod["spec"]["containers"][0]

    assert APPIO_CREDENTIALS_MOUNT_PATH == "/run/flwr/appio"
    assert pod["metadata"]["name"] == "flwr-taskexecutor-123"
    assert pod["metadata"]["namespace"] == "flower-system"
    assert container["image"] == "ghcr.io/flwrlabs/taskexecutor:dev"
    assert container["command"] == ["flwr-serverapp"]
    assert container["args"] == [
        "--serverappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]
    assert "task-token" not in container["command"]
    assert "task-token" not in container["args"]
    assert container["volumeMounts"] == [
        {
            "name": "appio-credentials",
            "mountPath": APPIO_CREDENTIALS_MOUNT_PATH,
            "readOnly": True,
        }
    ]
    assert pod["spec"]["volumes"] == [
        {
            "name": "appio-credentials",
            "secret": {
                "secretName": "flwr-taskexecutor-123-appio",
                "defaultMode": 0o444,
            },
        }
    ]
    assert pod["spec"]["automountServiceAccountToken"] is False
    assert pod["spec"]["restartPolicy"] == "Never"


def test_build_taskexecutor_pod_supports_clientapp_insecure_args() -> None:
    """Test Pod construction for insecure ClientApp launch args."""
    pod = build_taskexecutor_pod(
        _execution_spec(task_type=TaskType.CLIENT_APP, insecure=True),
        _executor_config(appio_root_certificates=None),
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-clientapp"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--clientappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--insecure",
    ]


def test_build_taskexecutor_pod_supports_secure_default_trust_store() -> None:
    """Test secure Pod args can rely on container default trust store."""
    config = _executor_config(appio_root_certificates=None)

    secret = build_appio_credentials_secret(_execution_spec(), config)
    pod = build_taskexecutor_pod(_execution_spec(), config)

    assert secret["stringData"] == {"token": "task-token"}
    assert pod["spec"]["containers"][0]["args"] == [
        "--serverappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
    ]


def test_build_taskexecutor_pod_supports_simulation_args() -> None:
    """Test Pod construction for Simulation launch args."""
    pod = build_taskexecutor_pod(
        _execution_spec(task_type=TaskType.SIMULATION),
        _executor_config(),
    )

    assert pod["spec"]["containers"][0]["command"] == ["flwr-simulation"]
    assert pod["spec"]["containers"][0]["args"] == [
        "--serverappio-api-address",
        "appio.example.com:9092",
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
        "--root-certificates",
        APPIO_ROOT_CERTIFICATES_FILE_PATH,
    ]


def test_build_taskexecutor_pod_supports_optional_container_config() -> None:
    """Test Pod construction includes optional Kubernetes container config."""
    pod = build_taskexecutor_pod(
        _execution_spec(runtime_dependency_install=True),
        _executor_config(
            image_pull_policy="IfNotPresent",
            service_account_name="flower-superexec",
        ),
    )
    container = pod["spec"]["containers"][0]

    assert container["imagePullPolicy"] == "IfNotPresent"
    assert "--allow-runtime-dependency-installation" in container["args"]
    assert pod["spec"]["serviceAccountName"] == "flower-superexec"


def test_build_taskexecutor_pod_supports_resources_and_placement() -> None:
    """Test Pod construction includes resource and placement inputs."""
    resources = {
        "requests": {"cpu": "500m", "memory": "1Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }
    node_selector = {"flower.ai/node-pool": "taskexecutors"}
    tolerations = [
        {
            "key": "flower.ai/taskexecutor",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]
    affinity: dict[str, Any] = {
        "podAntiAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": []
        }
    }

    pod = build_taskexecutor_pod(
        _execution_spec(),
        _executor_config(
            resources=resources,
            node_selector=node_selector,
            tolerations=tolerations,
            affinity=affinity,
            priority_class_name="taskexecutor-priority",
        ),
    )

    assert pod["spec"]["containers"][0]["resources"] == resources
    assert pod["spec"]["nodeSelector"] == node_selector
    assert pod["spec"]["tolerations"] == tolerations
    assert pod["spec"]["affinity"] == affinity
    assert pod["spec"]["priorityClassName"] == "taskexecutor-priority"


def test_build_taskexecutor_pod_supports_labels_annotations_and_security() -> None:
    """Test Pod construction includes object metadata and security fields."""
    pod_security_context = {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container_security_context = {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }
    config = _executor_config(
        labels={"flower.ai/team": "platform"},
        annotations={"flower.ai/owner": "superexec"},
        resource_pool="gpu-pool",
        pod_security_context=pod_security_context,
        container_security_context=container_security_context,
    )

    secret = build_appio_credentials_secret(_execution_spec(), config)
    pod = build_taskexecutor_pod(_execution_spec(), config)

    expected_labels = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": "123",
        "flower.ai/task-type": "flwr-serverapp",
        "flower.ai/resource-pool": "gpu-pool",
        "flower.ai/team": "platform",
    }
    assert secret["metadata"]["labels"] == expected_labels
    assert secret["metadata"]["annotations"] == {"flower.ai/owner": "superexec"}
    assert pod["metadata"]["labels"] == expected_labels
    assert pod["metadata"]["annotations"] == {"flower.ai/owner": "superexec"}
    assert pod["spec"]["securityContext"] == pod_security_context
    assert pod["spec"]["containers"][0]["securityContext"] == container_security_context


def test_build_taskexecutor_pod_never_exposes_token_in_container_spec() -> None:
    """Test task token is mounted by file and never in command, args, or env."""
    pod = build_taskexecutor_pod(_execution_spec(), _executor_config())
    container = pod["spec"]["containers"][0]

    assert "env" not in container
    assert not _contains_value(container["command"], "task-token")
    assert not _contains_value(container["args"], "task-token")


def test_config_rejects_extra_labels_that_override_stable_labels() -> None:
    """Test extra labels cannot replace executor-owned stable labels."""
    with pytest.raises(ValueError, match="must not override stable labels"):
        _executor_config(labels={"flower.ai/superexec-task-id": "999"})


def test_config_rejects_empty_annotation_entries() -> None:
    """Test annotation entries must be explicit non-empty strings."""
    with pytest.raises(ValueError, match="Kubernetes annotations"):
        _executor_config(annotations={"flower.ai/owner": ""})


def test_config_rejects_non_string_node_selector_entries() -> None:
    """Test node selector entries must be strings."""
    with pytest.raises(ValueError, match="Node selector entries must be strings"):
        _executor_config(node_selector={"flower.ai/gpu": True})


def test_config_rejects_capacity_budget_without_resource_pool() -> None:
    """Test capacity gating requires an explicit resource pool."""
    with pytest.raises(ValueError, match="Resource pool must be configured"):
        _executor_config(active_pod_budget=10)


def test_config_rejects_invalid_capacity_poll_interval() -> None:
    """Test capacity poll interval must be positive."""
    with pytest.raises(ValueError, match="Capacity poll interval must be positive"):
        _executor_config(capacity_poll_interval=0)


def test_wait_for_capacity_returns_below_budget_without_sleeping() -> None:
    """Test capacity wait returns immediately when the active Pod count fits."""
    client = Mock()
    client.list_namespaced_pod.return_value = {"items": [_pod("Running")]}
    sleep = Mock()
    config = _executor_config(
        labels={"flower.ai/team": "platform"},
        resource_pool="gpu-pool",
        active_pod_budget=2,
        capacity_poll_interval=3.0,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    client.list_namespaced_pod.assert_called_once_with(
        "flower-system",
        label_selector=(
            "app.kubernetes.io/component=taskexecutor,"
            "app.kubernetes.io/name=flower,"
            "flower.ai/resource-pool=gpu-pool,"
            "flower.ai/team=platform"
        ),
    )
    sleep.assert_not_called()


def test_wait_for_capacity_sleeps_and_polls_again_at_budget() -> None:
    """Test capacity wait sleeps when the active Pod count reaches the budget."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {"items": [_pod("Pending")]},
        {"items": []},
    ]
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        capacity_poll_interval=3.0,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    assert client.list_namespaced_pod.call_count == 2
    sleep.assert_called_once_with(3.0)


def test_wait_for_capacity_counts_pending_running_and_terminating_pods() -> None:
    """Test active Pod counting includes phase-active and terminating Pods."""
    client = Mock()
    client.list_namespaced_pod.side_effect = [
        {
            "items": [
                _pod("Pending"),
                _pod("Running"),
                _pod("Succeeded", deletion_timestamp="2026-05-26T18:30:00Z"),
                _pod("Failed", deletion_timestamp="2026-05-26T18:31:00Z"),
            ]
        },
        {"items": []},
    ]
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=4,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    sleep.assert_called_once_with(1.0)


def test_wait_for_capacity_ignores_terminal_pods_not_terminating() -> None:
    """Test Succeeded and Failed Pods do not count when they are not terminating."""
    client = Mock()
    client.list_namespaced_pod.return_value = {
        "items": [_pod("Succeeded"), _pod("Failed")]
    }
    sleep = Mock()
    config = _executor_config(
        resource_pool="gpu-pool",
        active_pod_budget=1,
        sleep=sleep,
    )

    KubernetesExecutor(client=client, config=config).wait_for_capacity()

    sleep.assert_not_called()


def test_launch_submits_secret_before_pod_and_returns_accepted() -> None:
    """Test launch creates the Secret before the Pod and returns accepted."""
    client = Mock()
    config = _executor_config()
    spec = _execution_spec()

    result = KubernetesExecutor(client=client, config=config).launch(spec)

    secret = build_appio_credentials_secret(spec, config)
    pod = build_taskexecutor_pod(spec, config)
    assert result.status == LaunchResultStatus.ACCEPTED
    assert client.mock_calls == [
        call.create_namespaced_secret("flower-system", secret),
        call.create_namespaced_pod("flower-system", pod),
    ]


def test_launch_returns_capacity_rejected_if_secret_create_hits_quota() -> None:
    """Test launch maps Secret quota rejection without creating the Pod."""
    client = Mock()
    client.create_namespaced_secret.side_effect = _KubernetesApiError(
        403, "exceeded quota: object-counts"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.CAPACITY_REJECTED
    assert result.message == "_KubernetesApiError: exceeded quota: object-counts"
    client.create_namespaced_pod.assert_not_called()


def test_launch_returns_capacity_rejected_if_pod_create_is_rate_limited() -> None:
    """Test launch maps Pod capacity rejection after Secret creation."""
    client = Mock()
    client.create_namespaced_pod.side_effect = _KubernetesApiError(
        429, "too many requests"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.CAPACITY_REJECTED
    assert result.message == "_KubernetesApiError: too many requests"
    client.create_namespaced_secret.assert_called_once()
    client.create_namespaced_pod.assert_called_once()


def test_launch_returns_failed_for_clear_non_capacity_failure() -> None:
    """Test launch maps clear non-capacity API failures to failed."""
    client = Mock()
    client.create_namespaced_secret.side_effect = _KubernetesApiError(
        401, "unauthorized"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.FAILED
    assert result.message == "_KubernetesApiError: unauthorized"
    client.create_namespaced_pod.assert_not_called()


def test_launch_returns_unknown_for_ambiguous_server_failure() -> None:
    """Test launch maps ambiguous server failures to unknown."""
    client = Mock()
    client.create_namespaced_pod.side_effect = _KubernetesApiError(
        503, "service unavailable"
    )

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec()
    )

    assert result.status == LaunchResultStatus.UNKNOWN
    assert result.message == "_KubernetesApiError: service unavailable"
    client.create_namespaced_secret.assert_called_once()
    client.create_namespaced_pod.assert_called_once()


def test_launch_returns_failed_if_spec_is_invalid() -> None:
    """Test launch returns failed for local spec validation errors."""
    client = Mock()

    result = KubernetesExecutor(client=client, config=_executor_config()).launch(
        _execution_spec(token="")
    )

    assert result.status == LaunchResultStatus.FAILED
    assert result.message == "ValueError: Kubernetes executor requires a task token."
    client.create_namespaced_secret.assert_not_called()
    client.create_namespaced_pod.assert_not_called()


def test_build_rejects_invalid_task_id() -> None:
    """Test Kubernetes object construction rejects invalid task IDs."""
    with pytest.raises(ValueError, match="positive integer task_id"):
        build_taskexecutor_pod(
            _execution_spec(task_id=0),
            _executor_config(),
        )
