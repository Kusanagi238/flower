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
"""Kubernetes executor for SuperExec TaskExecutor processes."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)

from .types import ExecutionSpec, LaunchResult

APPIO_CREDENTIALS_MOUNT_PATH = "/run/flwr/appio"
APPIO_TOKEN_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/token"
APPIO_ROOT_CERTIFICATES_FILE_PATH = f"{APPIO_CREDENTIALS_MOUNT_PATH}/ca.crt"
LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


class KubernetesClient(Protocol):
    """Subset of Kubernetes CoreV1Api used by the executor."""

    def create_namespaced_secret(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Secret in the selected namespace."""

    def create_namespaced_pod(self, namespace: str, body: dict[str, Any]) -> object:
        """Create a Kubernetes Pod in the selected namespace."""

    def list_namespaced_pod(self, namespace: str, label_selector: str) -> object:
        """List Kubernetes Pods in the selected namespace."""

    def list_namespaced_secret(self, namespace: str, label_selector: str) -> object:
        """List Kubernetes Secrets in the selected namespace."""

    def delete_namespaced_pod(
        self, name: str, namespace: str, grace_period_seconds: int = 0
    ) -> object:
        """Delete a Kubernetes Pod in the selected namespace."""

    def delete_namespaced_secret(self, name: str, namespace: str) -> object:
        """Delete a Kubernetes Secret in the selected namespace."""


@dataclass(frozen=True)
class KubernetesExecutorConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration needed to build one TaskExecutor Pod and Secret.

    appio_root_certificates contains optional PEM data mounted as ca.crt.
    """

    namespace: str
    image: str
    appio_root_certificates: str | None = None
    image_pull_policy: str | None = None
    labels: dict[str, str] | None = None
    annotations: dict[str, str] | None = None
    resource_pool: str | None = None
    resources: dict[str, Any] | None = None
    node_selector: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None
    affinity: dict[str, Any] | None = None
    priority_class_name: str | None = None
    pod_security_context: dict[str, Any] | None = None
    container_security_context: dict[str, Any] | None = None
    # Optional Pod field only; service account policy/RBAC is decided elsewhere.
    service_account_name: str | None = None
    active_pod_budget: int | None = None
    completed_pod_retention_seconds: float = 0.0
    capacity_poll_interval: float = 1.0
    capacity_log_interval: float | None = None
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = _utcnow

    def __post_init__(self) -> None:
        """Validate required object-building inputs."""
        if not self.namespace.strip():
            raise ValueError("Kubernetes namespace must not be empty.")
        if not self.image.strip():
            raise ValueError("TaskExecutor image must not be empty.")
        if self.appio_root_certificates is not None and not (
            self.appio_root_certificates.strip()
        ):
            raise ValueError("AppIo root certificates must not be empty.")
        if self.image_pull_policy is not None and not self.image_pull_policy.strip():
            raise ValueError("Image pull policy must not be empty.")
        if self.service_account_name is not None and not (
            self.service_account_name.strip()
        ):
            raise ValueError("Service account name must not be empty.")
        if self.resource_pool is not None and not self.resource_pool.strip():
            raise ValueError("Resource pool must not be empty.")
        if self.priority_class_name is not None and not (
            self.priority_class_name.strip()
        ):
            raise ValueError("Priority class name must not be empty.")
        if self.labels is not None:
            _validate_labels(self.labels)
        if self.annotations is not None:
            _validate_string_map("Kubernetes annotations", self.annotations)
        if self.node_selector is not None:
            _validate_string_map("Node selector", self.node_selector)
        if self.active_pod_budget is not None:
            if self.active_pod_budget <= 0:
                raise ValueError("Active Pod budget must be positive.")
            if self.resource_pool is None:
                raise ValueError(
                    "Resource pool must be configured when active Pod budget is set."
                )
        if self.capacity_poll_interval <= 0:
            raise ValueError("Capacity poll interval must be positive.")
        if self.capacity_log_interval is not None and self.capacity_log_interval <= 0:
            raise ValueError("Capacity log interval must be positive.")
        if self.completed_pod_retention_seconds < 0:
            raise ValueError("Completed Pod retention must not be negative.")


class KubernetesExecutor:
    """Submit TaskExecutor Pods to Kubernetes."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config

    def wait_for_capacity(self) -> None:
        """Wait until the configured resource pool is below its active Pod budget."""
        if self._config.active_pod_budget is None:
            return

        last_log_at: float | None = None
        while True:
            active_pod_count = self._active_pod_count()
            if active_pod_count < self._config.active_pod_budget:
                return

            if self._config.capacity_log_interval is not None:
                now = self._config.monotonic()
                if (
                    last_log_at is None
                    or now - last_log_at >= self._config.capacity_log_interval
                ):
                    LOGGER.info(
                        "Waiting for Kubernetes TaskExecutor capacity: "
                        "%s active Pods, budget %s, selector %s",
                        active_pod_count,
                        self._config.active_pod_budget,
                        _capacity_label_selector(self._config),
                    )
                    last_log_at = now

            self._config.sleep(self._config.capacity_poll_interval)

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Submit the TaskExecutor Pod and credential Secret."""
        try:
            secret = build_appio_credentials_secret(spec, self._config)
            pod = build_taskexecutor_pod(spec, self._config)
            self._client.create_namespaced_secret(self._config.namespace, secret)
            self._client.create_namespaced_pod(self._config.namespace, pod)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return _launch_result_from_exception(exc)

        return LaunchResult.accepted()

    def _active_pod_count(self) -> int:
        """Return the active TaskExecutor Pod count for the configured pool."""
        pod_list = self._client.list_namespaced_pod(
            self._config.namespace,
            label_selector=_capacity_label_selector(self._config),
        )
        return sum(1 for pod in _pod_items(pod_list) if _is_active_pod(pod))


class CompletedPodSweeper:
    """Delete terminal TaskExecutor Pods and orphaned credential Secrets."""

    def __init__(
        self,
        *,
        client: KubernetesClient,
        config: KubernetesExecutorConfig,
    ) -> None:
        self._client = client
        self._config = config

    def sweep(self) -> None:
        """Delete eligible terminal Pods and orphaned per-task Secrets."""
        selector = _taskexecutor_pool_label_selector(self._config)
        pods = _pod_items(
            self._client.list_namespaced_pod(
                self._config.namespace, label_selector=selector
            )
        )
        secrets = _secret_items(
            self._client.list_namespaced_secret(
                self._config.namespace, label_selector=selector
            )
        )
        remaining_pod_task_ids: set[str] = set()
        swept_pod_task_ids: set[str] = set()

        for pod in pods:
            task_id = _object_task_id(pod)
            if task_id is None:
                continue
            if _is_eligible_terminal_pod(pod, self._config):
                self._delete_pod(pod)
                swept_pod_task_ids.add(task_id)
            else:
                remaining_pod_task_ids.add(task_id)

        for secret in secrets:
            task_id = _object_task_id(secret)
            if task_id is None:
                continue
            if task_id in swept_pod_task_ids or task_id not in remaining_pod_task_ids:
                self._delete_secret(secret)

    def _delete_pod(self, pod: object) -> None:
        """Delete a Pod, tolerating already-deleted objects."""
        name = _object_name(pod)
        if name is None:
            return
        try:
            self._client.delete_namespaced_pod(
                name=name,
                namespace=self._config.namespace,
                grace_period_seconds=0,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)

    def _delete_secret(self, secret: object) -> None:
        """Delete a Secret, tolerating already-deleted objects."""
        name = _object_name(secret)
        if name is None:
            return
        try:
            self._client.delete_namespaced_secret(
                name=name, namespace=self._config.namespace
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _raise_unless_not_found(exc)


def build_appio_credentials_secret(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> dict[str, Any]:
    """Build the AppIo credential Secret for a TaskExecutor Pod."""
    _validate_kubernetes_spec(spec)
    data = {"token": spec.token}
    if config.appio_root_certificates is not None:
        data["ca.crt"] = config.appio_root_certificates

    return {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": _metadata(_credential_secret_name(spec), spec, config),
        "type": "Opaque",
        "stringData": data,
    }


def build_taskexecutor_pod(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> dict[str, Any]:
    """Build the TaskExecutor Pod for a claimed SuperExec task."""
    _validate_kubernetes_spec(spec)

    container: dict[str, Any] = {
        "name": "taskexecutor",
        "image": config.image,
        "command": [TASK_TYPE_TO_COMMAND[spec.task_type]],
        "args": _taskexecutor_args(spec, config),
        "volumeMounts": [
            {
                "name": "appio-credentials",
                "mountPath": APPIO_CREDENTIALS_MOUNT_PATH,
                "readOnly": True,
            }
        ],
    }
    if config.image_pull_policy is not None:
        container["imagePullPolicy"] = config.image_pull_policy
    if config.resources is not None:
        container["resources"] = config.resources
    if config.container_security_context is not None:
        container["securityContext"] = config.container_security_context

    pod_spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": [
            {
                "name": "appio-credentials",
                "secret": {
                    "secretName": _credential_secret_name(spec),
                    "defaultMode": 0o444,
                },
            }
        ],
    }
    if config.service_account_name is not None:
        pod_spec["serviceAccountName"] = config.service_account_name
    if config.node_selector is not None:
        pod_spec["nodeSelector"] = config.node_selector
    if config.tolerations is not None:
        pod_spec["tolerations"] = config.tolerations
    if config.affinity is not None:
        pod_spec["affinity"] = config.affinity
    if config.priority_class_name is not None:
        pod_spec["priorityClassName"] = config.priority_class_name
    if config.pod_security_context is not None:
        pod_spec["securityContext"] = config.pod_security_context

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": _metadata(_pod_name(spec), spec, config),
        "spec": pod_spec,
    }


def _taskexecutor_args(
    spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> list[str]:
    """Build TaskExecutor arguments with file-based credential delivery."""
    args = [
        TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
        spec.appio_api_address,
        "--token-file",
        APPIO_TOKEN_FILE_PATH,
    ]

    if spec.insecure:
        args.append("--insecure")
    elif config.appio_root_certificates is not None:
        args.extend(["--root-certificates", APPIO_ROOT_CERTIFICATES_FILE_PATH])

    if spec.runtime_dependency_install:
        args.append("--allow-runtime-dependency-installation")

    return args


def _validate_kubernetes_spec(spec: ExecutionSpec) -> None:
    """Validate spec inputs required for Kubernetes object construction."""
    if not isinstance(spec.task_id, int) or spec.task_id <= 0:
        raise ValueError("Kubernetes executor requires a positive integer task_id.")
    if not spec.appio_api_address.strip():
        raise ValueError("Kubernetes executor requires an AppIo API address.")
    if not spec.token.strip():
        raise ValueError("Kubernetes executor requires a task token.")


def _pod_name(spec: ExecutionSpec) -> str:
    """Return the TaskExecutor Pod name."""
    return f"flwr-taskexecutor-{spec.task_id}"


def _credential_secret_name(spec: ExecutionSpec) -> str:
    """Return the AppIo credential Secret name."""
    return f"{_pod_name(spec)}-appio"


def _metadata(
    name: str, spec: ExecutionSpec, config: KubernetesExecutorConfig
) -> dict[str, Any]:
    """Return Kubernetes object metadata."""
    metadata: dict[str, Any] = {
        "name": name,
        "namespace": config.namespace,
        "labels": _labels(spec, config),
    }
    if config.annotations is not None:
        metadata["annotations"] = config.annotations
    return metadata


def _labels(spec: ExecutionSpec, config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return stable labels for Kubernetes objects."""
    labels = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
        "flower.ai/superexec-task-id": str(spec.task_id),
        "flower.ai/task-type": spec.task_type.value,
    }
    if config.resource_pool is not None:
        labels["flower.ai/resource-pool"] = config.resource_pool
    if config.labels is not None:
        labels.update(config.labels)
    return labels


def _capacity_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector used for resource-pool capacity checks."""
    return _taskexecutor_pool_label_selector(config)


def _taskexecutor_pool_label_selector(config: KubernetesExecutorConfig) -> str:
    """Return the label selector for TaskExecutor pool-scoped operations."""
    return _label_selector(_taskexecutor_pool_labels(config))


def _taskexecutor_pool_labels(config: KubernetesExecutorConfig) -> dict[str, str]:
    """Return labels identifying a scoped TaskExecutor pool."""
    labels = {
        "app.kubernetes.io/name": "flower",
        "app.kubernetes.io/component": "taskexecutor",
    }
    if config.resource_pool is not None:
        labels["flower.ai/resource-pool"] = config.resource_pool
    if config.labels is not None:
        labels.update(config.labels)
    return labels


def _label_selector(labels: dict[str, str]) -> str:
    """Return a Kubernetes equality label selector."""
    return ",".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _pod_items(pod_list: object) -> list[Any]:
    """Return Pod items from a Kubernetes list response."""
    items = _object_field(pod_list, "items")
    if isinstance(items, list):
        return items
    return []


def _secret_items(secret_list: object) -> list[Any]:
    """Return Secret items from a Kubernetes list response."""
    items = _object_field(secret_list, "items")
    if isinstance(items, list):
        return items
    return []


def _is_active_pod(pod: object) -> bool:
    """Return true if a Pod counts against best-effort launch capacity."""
    metadata = _object_field(pod, "metadata")
    deletion_timestamp = _object_field(metadata, "deletion_timestamp")
    if deletion_timestamp is None:
        deletion_timestamp = _object_field(metadata, "deletionTimestamp")
    if deletion_timestamp is not None:
        return True

    status = _object_field(pod, "status")
    return _object_field(status, "phase") in {"Pending", "Running"}


def _is_eligible_terminal_pod(pod: object, config: KubernetesExecutorConfig) -> bool:
    """Return true if a terminal Pod is old enough for cleanup."""
    status = _object_field(pod, "status")
    if _object_field(status, "phase") not in {"Succeeded", "Failed"}:
        return False

    if config.completed_pod_retention_seconds == 0:
        return True

    terminal_time = _terminal_time(pod)
    if terminal_time is None:
        # Without a termination timestamp, a configured retention window cannot
        # be evaluated safely, so keep the Pod for a later sweep.
        return False

    elapsed = _as_utc(config.now()) - terminal_time
    return elapsed.total_seconds() >= config.completed_pod_retention_seconds


def _terminal_time(pod: object) -> datetime | None:
    """Return the latest container termination time for a Pod."""
    status = _object_field(pod, "status")
    container_statuses = _object_field(status, "container_statuses")
    if container_statuses is None:
        container_statuses = _object_field(status, "containerStatuses")
    if not isinstance(container_statuses, list):
        return None

    terminal_times = []
    for container_status in container_statuses:
        state = _object_field(container_status, "state")
        terminated = _object_field(state, "terminated")
        if terminated is None:
            continue
        finished_at = _object_field(terminated, "finished_at")
        if finished_at is None:
            finished_at = _object_field(terminated, "finishedAt")
        terminal_time = _parse_datetime(finished_at)
        if terminal_time is not None:
            terminal_times.append(terminal_time)
    if not terminal_times:
        return None
    return max(terminal_times)


def _parse_datetime(value: object) -> datetime | None:
    """Parse Kubernetes timestamp values."""
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str):
        return None
    timestamp = value.strip()
    if not timestamp:
        return None
    if timestamp.endswith("Z"):
        timestamp = f"{timestamp[:-1]}+00:00"
    try:
        return _as_utc(datetime.fromisoformat(timestamp))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _object_name(value: object) -> str | None:
    """Return an object's metadata name."""
    metadata = _object_field(value, "metadata")
    name = _object_field(metadata, "name")
    if isinstance(name, str) and name.strip():
        return name
    return None


def _object_task_id(value: object) -> str | None:
    """Return an object's stable TaskExecutor task-id label."""
    metadata = _object_field(value, "metadata")
    labels = _object_field(metadata, "labels")
    if not isinstance(labels, dict):
        return None
    task_id = labels.get("flower.ai/superexec-task-id")
    if isinstance(task_id, str) and task_id.strip():
        return task_id
    return None


def _object_field(value: object, field_name: str) -> object | None:
    """Return a field from a Kubernetes dict or model object."""
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _validate_labels(labels: dict[str, str]) -> None:
    """Validate that caller-provided labels do not replace stable labels."""
    stable_label_names = {
        "app.kubernetes.io/name",
        "app.kubernetes.io/component",
        "flower.ai/superexec-task-id",
        "flower.ai/task-type",
        "flower.ai/resource-pool",
    }
    conflicts = sorted(stable_label_names.intersection(labels))
    if conflicts:
        raise ValueError(
            f"Kubernetes labels must not override stable labels: {conflicts}"
        )
    _validate_string_map("Kubernetes labels", labels)


def _validate_string_map(name: str, values: dict[str, str]) -> None:
    """Validate a non-empty string mapping."""
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(f"{name} entries must be strings.")
        if not key.strip() or not value.strip():
            raise ValueError(f"{name} entries must not be empty.")


def _launch_result_from_exception(exc: Exception) -> LaunchResult:
    """Map immediate Kubernetes API exceptions to launch results."""
    message = f"{type(exc).__name__}: {exc}"
    status = _exception_status(exc)
    lower_message = message.lower()

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return LaunchResult.unknown(message)

    if status == 429 or _is_capacity_message(lower_message):
        return LaunchResult.capacity_rejected(message)

    if status is not None and (status == 408 or status >= 500):
        return LaunchResult.unknown(message)

    return LaunchResult.failed(message)


def _exception_status(exc: Exception) -> int | None:
    """Return an HTTP-like status from Kubernetes client exceptions."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if isinstance(status, str) and status.isdigit():
        return int(status)
    return None


def _raise_unless_not_found(exc: Exception) -> None:
    """Raise Kubernetes client exceptions except already-deleted objects."""
    if _exception_status(exc) == 404:
        return
    raise exc


def _is_capacity_message(message: str) -> bool:
    """Return true for quota/admission capacity rejection messages."""
    capacity_markers = (
        "exceeded quota",
        "resourcequota",
        "quota exceeded",
        "too many requests",
        "rate limit",
        "insufficient cpu",
        "insufficient memory",
        "insufficient pods",
    )
    return any(marker in message for marker in capacity_markers)
