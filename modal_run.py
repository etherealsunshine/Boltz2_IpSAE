#need to fix this to accomodate new ways for input, currently it works only for the existing yaml method


import modal
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil
app = modal.App("boltz2-ipsae")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "biopython==1.84",
        "numpy==1.26.3",
        "pyyaml>=6.0",
        "torch",  
    )
    .pip_install(
        "boltz>=2.2.1",  
    )
    .run_commands(
        #cuequvariance was giving some troubles, need to clean this up eventually
        "pip install cuequivariance-torch || pip install cuequivariance_torch || echo 'Warning: Could not install cuequivariance-torch automatically'"
    )
    .add_local_file("ipsae.py", "/root/ipsae.py")  # Add ipsae.py script to image
)


@app.function(
    image=image,
    gpu="A10G",  # GPU options: A10G() (faster), T4() (cheaper), or None for CPU
    timeout=3600,  # 1 hour timeout (adjust as needed)
    volumes={
        "/data": modal.Volume.from_name("boltz-data", create_if_missing=True),
    },
)
def run_boltz_predict(
    yaml_content: str,
    yaml_filename: str,
    msa_files: Optional[Dict[str, bytes]] = None,
    recycling_steps: int = 10,
    diffusion_samples: int = 5,
    use_msa_server: bool = False,
    output_dir: str = "/data/outputs",
) -> Dict[str, Any]:
    """
    Run boltz predict on Modal GPU.
    
    Args:
        yaml_content: Content of the YAML config file
        yaml_filename: Name for the YAML file
        msa_files: Optional dict of {filename: content_bytes} for MSA files (e.g., .a3m files)
        recycling_steps: Number of recycling steps
        diffusion_samples: Number of diffusion samples
        use_msa_server: Whether to use MSA server
        output_dir: Directory to save outputs
    
    Returns:
        Dict with status and output paths
    """
    
    
    # Create working directory
    work_dir = Path("/tmp/boltz_work")
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Write YAML file
        yaml_path = work_dir / yaml_filename
        yaml_path.write_text(yaml_content)
        
        # Write MSA files if provided
        if msa_files:
            for msa_filename, msa_content in msa_files.items():
                msa_path = work_dir / msa_filename
                if isinstance(msa_content, bytes):
                    msa_path.write_bytes(msa_content)
                else:
                    msa_path.write_text(msa_content)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Build boltz command
        cmd = [
            "boltz", "predict",
            str(yaml_path),
            "--recycling_steps", str(recycling_steps),
            "--diffusion_samples", str(diffusion_samples),
            "--out_dir", str(output_path),
            "--no_kernels",  # Avoid cuequivariance_torch dependency, issue on github says this is only way to ensure it works
        ]
        
        if use_msa_server:
            cmd.append("--use_msa_server")
        
        # Change to work directory (so relative paths in YAML work)
        os.chdir(work_dir)
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {work_dir}")
        print(f"Output directory: {output_path}")
        
        # Run boltz predict
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # List output files
        output_files = []
        if output_path.exists():
            for file_path in output_path.rglob("*"):
                if file_path.is_file():
                    output_files.append(str(file_path.relative_to(output_path)))
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_dir": str(output_path),
            "output_files": sorted(output_files),
        }
        
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)


@app.local_entrypoint()
def main(
    yaml_file: str = "example_yaml/boltz_validation/binder_2vsm/binder_2vsm_vs_target_nipah_g.yaml",
):
    """
    Example usage: Run a single Boltz2 prediction.
    
    Usage:
        modal run get_started.py::main
        modal run get_started.py::main --yaml-file example_yaml/boltz_validation/binder_2vsm/binder_2vsm_vs_target_nipah_g.yaml
    """
    
    
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {yaml_file}")
        return
    
    # Read YAML content
    yaml_content = yaml_path.read_text()
    yaml_filename = yaml_path.name
    
    # Read MSA files if they exist
    msa_files = {}
    msa_dir = yaml_path.parent.parent.parent  # Go up to example_yaml
    nipah_msa = msa_dir / "nipah.a3m"
    if nipah_msa.exists():
        msa_files["nipah.a3m"] = nipah_msa.read_bytes()
        print(f"Found MSA file: nipah.a3m")
    
    print(f"Running Boltz2 prediction for: {yaml_file}")
    result = run_boltz_predict.remote(
        yaml_content=yaml_content,
        yaml_filename=yaml_filename,
        msa_files=msa_files if msa_files else None,
        recycling_steps=10,
        diffusion_samples=5,
        use_msa_server=False,
    )
    
    print("\n" + "="*50)
    print("RESULTS:")
    print("="*50)
    print(f"Status: {result['status']}")
    print(f"Return code: {result['returncode']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"\nOutput files ({len(result['output_files'])}):")
    for f in result['output_files'][:10]:  # Show first 10
        print(f"  - {f}")
    if len(result['output_files']) > 10:
        print(f"  ... and {len(result['output_files']) - 10} more")
    
    if result['stderr']:
        print(f"\nSTDERR:\n{result['stderr']}")
    
    # Note: To download files, you would need to use Modal volumes or download them
    print("\nNote: Outputs are stored in Modal volume. Use Modal CLI to download if needed.")


@app.local_entrypoint()
def run_batch(
    binder_dir: str = "example_yaml/boltz_validation/binder_2vsm",
    msa_dir: str = "example_yaml",
):
    """
    Run all YAML files in a binder directory.
    
    Usage:
        modal run get_started.py::run_batch --binder-dir example_yaml/boltz_validation/binder_2vsm
    """
    from pathlib import Path
    
    binder_path = Path(binder_dir)
    msa_path = Path(msa_dir)
    
    # Find all YAML files
    yaml_files = list(binder_path.glob("*.yaml"))
    
    if not yaml_files:
        print(f"No YAML files found in {binder_dir}")
        return
    
    print(f"Found {len(yaml_files)} YAML files to process")
    
    # Read MSA files
    msa_files = {}
    nipah_msa = msa_path / "nipah.a3m"
    if nipah_msa.exists():
        msa_files["nipah.a3m"] = nipah_msa.read_bytes()
        print(f"Found MSA file: nipah.a3m")
    
    results = []
    for yaml_file in yaml_files:
        print(f"\nProcessing: {yaml_file.name}")
        try:
            yaml_content = yaml_file.read_text()
            result = run_boltz_predict.remote(
                yaml_content=yaml_content,
                yaml_filename=yaml_file.name,
                msa_files=msa_files if msa_files else None,
                recycling_steps=10,
                diffusion_samples=5,
                use_msa_server=False,
            )
            results.append((yaml_file.name, result))
            print(f"  ✓ Completed: {result['status']}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append((yaml_file.name, {"status": "error", "error": str(e)}))
    
    print("\n" + "="*50)
    print("BATCH RESULTS SUMMARY:")
    print("="*50)
    for name, result in results:
        status = result.get("status", "unknown")
        print(f"{name}: {status}")


@app.function(
    image=image,
    volumes={
        "/data": modal.Volume.from_name("boltz-data", create_if_missing=True),
    },
)
def calculate_ipsae(
    prediction_dir: str,
    pae_cutoff: float = 15.0,
    dist_cutoff: float = 15.0,
    model_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate ipSAE scores from Boltz prediction outputs.
    
    Args:
        prediction_dir: Path to predictions directory (e.g., "boltz_results_binder_2vsm_vs_target_nipah_g/predictions/binder_2vsm_vs_target_nipah_g")
        pae_cutoff: PAE cutoff value (default 15.0)
        dist_cutoff: Distance cutoff value (default 15.0)
        model_number: Specific model number (0-4), or None to process all models
    
    Returns:
        Dict with ipSAE scores for each model
    """
    from pathlib import Path
    import subprocess
    import json
    import re
    
    base_dir = Path("/data/outputs") / prediction_dir
    if not base_dir.exists():
        return {"error": f"Directory not found: {base_dir}"}
    
    results = {}
    
    # Determine which models to process
    if model_number is not None:
        model_numbers = [model_number]
    else:
        # Find all available models
        cif_files = list(base_dir.glob("*_model_*.cif"))
        model_numbers = []
        for cif_file in cif_files:
            match = re.search(r"_model_(\d+)\.cif", cif_file.name)
            if match:
                model_numbers.append(int(match.group(1)))
        model_numbers = sorted(set(model_numbers))
    
    if not model_numbers:
        return {"error": "No model files found"}
    
    for model_num in model_numbers:
        # Find PAE and CIF files for this model
        pae_files = list(base_dir.glob(f"*_model_{model_num}.npz"))
        cif_files = list(base_dir.glob(f"*_model_{model_num}.cif"))
        
        # Find the PAE file (should be pae_*.npz)
        pae_file = None
        for pf in pae_files:
            if "pae" in pf.name.lower():
                pae_file = pf
                break
        
        # Find the CIF file
        cif_file = cif_files[0] if cif_files else None
        
        if not pae_file or not cif_file:
            results[f"model_{model_num}"] = {
                "error": f"Missing files: pae={pae_file is None}, cif={cif_file is None}"
            }
            continue
        
        # Run ipsae.py
        cmd = [
            "python", "/root/ipsae.py",
            str(pae_file),
            str(cif_file),
            str(pae_cutoff),
            str(dist_cutoff),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(base_dir),
                check=True,
            )
            
            # Find the output file
            # ipsae.py creates: {cif_stem}_{pae_cutoff}_{dist_cutoff}.txt
            # Format pae_cutoff and dist_cutoff with leading zeros if < 10
            pae_str = f"{int(pae_cutoff):02d}" if pae_cutoff < 10 else str(int(pae_cutoff))
            dist_str = f"{int(dist_cutoff):02d}" if dist_cutoff < 10 else str(int(dist_cutoff))
            cif_stem = cif_file.stem  # Get filename without extension
            output_file = cif_file.parent / f"{cif_stem}_{pae_str}_{dist_str}.txt"
            
            if output_file.exists():
                # Parse the output file to extract ipSAE scores
                with open(output_file, 'r') as f:
                    lines = f.readlines()
                
                # Find the header line to understand column positions
                header_line = None
                for i, line in enumerate(lines):
                    if 'Chn1' in line and 'ipSAE' in line:
                        header_line = line
                        break
                
                # Parse data lines
                ipSAE_values = []
                ipSAE_d0chn_values = []
                ipSAE_d0dom_values = []
                chain_pairs = []
                
                data_started = False
                for line in lines:
                    # Skip until we find the header
                    if header_line and not data_started:
                        if 'Chn1' in line and 'ipSAE' in line:
                            data_started = True
                            continue
                        continue
                    
                    if not data_started:
                        continue
                    
                    # Skip empty lines and comments
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse data lines - split on whitespace
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            chn1 = parts[0]
                            chn2 = parts[1]
                            # Column indices: Chn1 Chn2 PAE Dist Type ipSAE ipSAE_d0chn ipSAE_d0dom ...
                            if len(parts) > 5:
                                ipSAE_val = float(parts[5])
                                ipSAE_values.append(ipSAE_val)
                                chain_pairs.append(f"{chn1}-{chn2}")
                            if len(parts) > 6:
                                ipSAE_d0chn_val = float(parts[6])
                                ipSAE_d0chn_values.append(ipSAE_d0chn_val)
                            if len(parts) > 7:
                                ipSAE_d0dom_val = float(parts[7])
                                ipSAE_d0dom_values.append(ipSAE_d0dom_val)
                        except (ValueError, IndexError) as e:
                            continue
                
                # Get statistics
                ipSAE_min = min(ipSAE_values) if ipSAE_values else None
                ipSAE_max = max(ipSAE_values) if ipSAE_values else None
                ipSAE_mean = sum(ipSAE_values) / len(ipSAE_values) if ipSAE_values else None
                
                results[f"model_{model_num}"] = {
                    "status": "success",
                    "pae_file": str(pae_file.name),
                    "cif_file": str(cif_file.name),
                    "ipSAE_min": ipSAE_min,
                    "ipSAE_max": ipSAE_max,
                    "ipSAE_mean": ipSAE_mean,
                    "ipSAE_d0chn_mean": sum(ipSAE_d0chn_values) / len(ipSAE_d0chn_values) if ipSAE_d0chn_values else None,
                    "ipSAE_d0dom_mean": sum(ipSAE_d0dom_values) / len(ipSAE_d0dom_values) if ipSAE_d0dom_values else None,
                    "chain_pairs": chain_pairs,
                    "num_chain_pairs": len(ipSAE_values),
                    "output_file": str(output_file.name),
                }
            else:
                results[f"model_{model_num}"] = {
                    "status": "error",
                    "error": f"Output file not created: {output_file}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except subprocess.CalledProcessError as e:
            results[f"model_{model_num}"] = {
                "status": "error",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
            }
    
    return results


@app.local_entrypoint()
def get_ipsae_scores(
    prediction_dir: str = "boltz_results_binder_2vsm_vs_target_nipah_g/predictions/binder_2vsm_vs_target_nipah_g",
    pae_cutoff: float = 15.0,
    dist_cutoff: float = 15.0,
    model_number: Optional[int] = None,
):
    """
    Calculate and display ipSAE scores for Boltz predictions.
    
    Usage:
        modal run get_started.py::get_ipsae_scores
        modal run get_started.py::get_ipsae_scores --prediction-dir "boltz_results_binder_2vsm_vs_target_nipah_g/predictions/binder_2vsm_vs_target_nipah_g" --model-number 0
    """
    results = calculate_ipsae.remote(
        prediction_dir=prediction_dir,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        model_number=model_number,
    )
    
    print("\n" + "="*60)
    print("ipSAE SCORES")
    print("="*60)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    for model_key, model_results in sorted(results.items()):
        print(f"\n{model_key.upper()}:")
        if model_results.get("status") == "success":
            print(f"  PAE file: {model_results['pae_file']}")
            print(f"  CIF file: {model_results['cif_file']}")
            print(f"  Chain pairs: {', '.join(model_results.get('chain_pairs', []))}")
            if model_results.get("ipSAE_min") is not None:
                print(f"  ipSAE_min: {model_results['ipSAE_min']:.6f}")
            if model_results.get("ipSAE_max") is not None:
                print(f"  ipSAE_max: {model_results['ipSAE_max']:.6f}")
            if model_results.get("ipSAE_mean") is not None:
                print(f"  ipSAE_mean: {model_results['ipSAE_mean']:.6f}")
            if model_results.get("ipSAE_d0chn_mean") is not None:
                print(f"  ipSAE_d0chn_mean: {model_results['ipSAE_d0chn_mean']:.6f}")
            if model_results.get("ipSAE_d0dom_mean") is not None:
                print(f"  ipSAE_d0dom_mean: {model_results['ipSAE_d0dom_mean']:.6f}")
            print(f"  Output file: {model_results['output_file']}")
        else:
            print(f"  Error: {model_results.get('error', 'Unknown error')}")
    
    print("\n" + "="*60)