from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS_ROOT = REPO_ROOT / "tools/phantom_gpu"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_with_private_umask(
    arguments: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'umask 077; exec "$@"', "evidence-test", *arguments],
        check=False,
        capture_output=True,
        text=True,
        **kwargs,
    )


def make_packaging_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    fixture_root = tmp_path / "repo"
    tools_root = fixture_root / "tools/phantom_gpu"
    tools_root.mkdir(parents=True)
    package_script = tools_root / "package_a100_health.sh"
    shutil.copy2(TOOLS_ROOT / "package_a100_health.sh", package_script)
    shutil.copy2(
        TOOLS_ROOT / "run_a100_health.sh", tools_root / "run_a100_health.sh"
    )

    run_id = "20260807T000000Z-1.abc123"
    results_root = fixture_root / "build/phantom_gpu/compile_only_results"
    run_root = results_root / "runs" / run_id
    codegen_root = run_root / "ckks2c"
    codegen_root.mkdir(parents=True)

    health_binary = codegen_root / "native_phantom_health_sm80"
    health_binary.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    health_binary.chmod(0o755)
    health_digest = sha256(health_binary)

    context_manifest = codegen_root / "compiler_context_manifest.json"
    resource_manifest = codegen_root / "compiler_resource_manifest.json"
    write_json(context_manifest, {"schema_version": 1})
    write_json(resource_manifest, {"schema_version": 1})
    context_digest = sha256(context_manifest)
    resource_digest = sha256(resource_manifest)
    write_json(
        codegen_root / "qualification.json",
        {
            "status": "pass",
            "executable_was_run": False,
            "health_binary_sha256": health_digest,
            "compiler_context_manifest_sha256": context_digest,
            "compiler_resource_manifest_sha256": resource_digest,
        },
    )

    sums_path = run_root / "SHA256SUMS"
    qualified_files = [
        health_binary,
        codegen_root / "qualification.json",
        context_manifest,
        resource_manifest,
    ]
    sums_path.write_text(
        "".join(
            f"{sha256(path)}  ./{path.relative_to(run_root)}\n"
            for path in qualified_files
        ),
        encoding="utf-8",
    )

    image_id = "sha256:" + "1" * 64
    definition_digest = "2" * 64
    manifest_path = run_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "status": "pass",
            "gate": "ckks2c",
            "run_id": run_id,
            "development_image_id": image_id,
            "development_definition_sha256": definition_digest,
            "evidence_sha256_manifest_sha256": sha256(sums_path),
            "compiler_context_manifest_sha256": context_digest,
        },
    )
    qualification_record = results_root / "current-ckks2c.json"
    write_json(
        qualification_record,
        {
            "status": "pass",
            "gate": "ckks2c",
            "exit_code": 0,
            "run_id": run_id,
            "run_root": str(run_root),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "development_image_id": image_id,
            "development_definition_sha256": definition_digest,
        },
    )
    return package_script, qualification_record, fixture_root, run_id


def test_packaged_health_archive_is_readable_under_private_umask(
    tmp_path: Path,
) -> None:
    package_script, qualification_record, fixture_root, run_id = (
        make_packaging_fixture(tmp_path)
    )
    output_root = fixture_root / "bundles"
    registry_image = "registry.invalid/health@sha256:" + "3" * 64

    result = run_with_private_umask(
        [
            "bash",
            str(package_script),
            "--qualification-record",
            str(qualification_record),
            "--registry-image",
            registry_image,
            "--output-root",
            str(output_root),
        ]
    )
    assert result.returncode == 0, result.stderr

    archive_path = output_root / f"{run_id}.tar.gz"
    sidecar_path = output_root / f"{run_id}.tar.gz.sha256"
    assert archive_path.stat().st_mode & 0o777 == 0o644
    assert sidecar_path.stat().st_mode & 0o777 == 0o644
    assert sidecar_path.read_text(encoding="utf-8").split()[0] == sha256(
        archive_path
    )
    with tarfile.open(archive_path, "r:gz") as archive:
        assert f"{run_id}/bundle_manifest.json" in archive.getnames()
    assert not list(output_root.glob(".*.tar.gz.*"))


def test_failed_health_result_archive_is_readable_under_private_umask(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    expected_image_id = "sha256:" + "4" * 64
    write_json(
        bundle_dir / "bundle_manifest.json",
        {
            "development_image_id": expected_image_id,
            "development_definition_sha256": "5" * 64,
            "compiler_context_manifest_sha256": "6" * 64,
            "health_binary_sha256": "7" * 64,
            "registry_image": "registry.invalid/health@sha256:" + "8" * 64,
            "health_timeout_seconds": 300,
        },
    )
    (bundle_dir / "SHA256SUMS").write_text(
        f"{sha256(bundle_dir / 'bundle_manifest.json')}  "
        "./bundle_manifest.json\n",
        encoding="utf-8",
    )
    result_dir = tmp_path / "result"
    archive_path = tmp_path / "result.tar.gz"
    environment = os.environ.copy()
    environment.update(
        {
            "ACE_PHANTOM_IMAGE_ID": "sha256:" + "9" * 64,
            "ACE_PHANTOM_DEFINITION_SHA256": "5" * 64,
            "ACE_PHANTOM_REGISTRY_IMAGE": (
                "registry.invalid/health@sha256:" + "8" * 64
            ),
        }
    )

    result = run_with_private_umask(
        [
            "bash",
            str(TOOLS_ROOT / "run_a100_health.sh"),
            "--bundle-dir",
            str(bundle_dir),
            "--result-dir",
            str(result_dir),
            "--archive",
            str(archive_path),
        ],
        env=environment,
    )
    assert result.returncode == 1
    assert archive_path.stat().st_mode & 0o777 == 0o644
    assert json.loads((result_dir / "state.json").read_text(encoding="utf-8")) == {
        "status": "failed",
        "exit_code": 1,
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        archived_state = json.load(archive.extractfile("result/state.json"))
    assert archived_state == {"status": "failed", "exit_code": 1}


def test_health_runner_reads_the_explicit_json_artifact_not_stdout_tail(
    tmp_path: Path,
) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    image_id = "sha256:" + "4" * 64
    definition_sha = "5" * 64
    context_sha = "6" * 64
    registry_image = "registry.invalid/health@sha256:" + "8" * 64
    health_binary = bundle_dir / "native_phantom_health_sm80"
    health_binary.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' "
        "'{\"status\":\"pass\",\"gpu\":\"NVIDIA A100 80GB PCIe\",'"
        "'\"device_count\":1,\"max_error\":0.000001,'"
        f"'\"context_manifest_sha256\":\"{context_sha}\"}}' >\"$1\"\n"
        "echo 'health diagnostic before footer'\n"
        "echo 'non-JSON footer'\n",
        encoding="utf-8",
    )
    health_binary.chmod(0o755)
    write_json(
        bundle_dir / "bundle_manifest.json",
        {
            "development_image_id": image_id,
            "development_definition_sha256": definition_sha,
            "compiler_context_manifest_sha256": context_sha,
            "health_binary_sha256": sha256(health_binary),
            "registry_image": registry_image,
            "health_timeout_seconds": 300,
        },
    )
    (bundle_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  ./{path.name}\n"
            for path in (
                bundle_dir / "bundle_manifest.json",
                health_binary,
            )
        ),
        encoding="utf-8",
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == -L ]]; then\n"
        "  echo 'GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-test)'\n"
        "else\n"
        "  echo 'NVIDIA A100 80GB PCIe, GPU-test, 81920 MiB, 550.90, P0, 250 W'\n"
        "fi\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "ACE_PHANTOM_IMAGE_ID": image_id,
            "ACE_PHANTOM_DEFINITION_SHA256": definition_sha,
            "ACE_PHANTOM_REGISTRY_IMAGE": registry_image,
        }
    )
    result_dir = tmp_path / "result"
    archive_path = tmp_path / "result.tar.gz"
    result = run_with_private_umask(
        [
            "bash",
            str(TOOLS_ROOT / "run_a100_health.sh"),
            "--bundle-dir",
            str(bundle_dir),
            "--result-dir",
            str(result_dir),
            "--archive",
            str(archive_path),
        ],
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert "non-JSON footer" in (
        result_dir / "native_health.stdout.txt"
    ).read_text(encoding="utf-8")
    assert json.loads(
        (result_dir / "native_health.json").read_text(encoding="utf-8")
    )["status"] == "pass"
    assert json.loads((result_dir / "state.json").read_text(encoding="utf-8")) == {
        "status": "pass",
        "exit_code": 0,
    }
