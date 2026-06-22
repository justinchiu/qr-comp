# Remote GPU Runbook

Use this runbook for rented CUDA iteration when this Mac is not enough. It
covers Prime Intellect and AWS EC2 Spot, then gives one shared A100/H100/B200
benchmarking and Nsight Compute workflow.

The leaderboard target remains Popcorn B200. A100 and H100 runs are useful for
correctness, launch/debug iteration, timing sweeps, and some profiler evidence,
but do not choose final B200 dispatch solely from A100 or H100 results.

## Source Docs

Prime:

- Prime CLI overview: https://docs.primeintellect.ai/cli-reference/introduction
- Prime CLI provisioning: https://docs.primeintellect.ai/cli-reference/provision-gpu
- Prime API availability: https://docs.primeintellect.ai/api-reference/check-gpu-availability
- Prime API provisioning: https://docs.primeintellect.ai/api-reference/provision-gpu
- Prime FAQ on spot instances: https://docs.primeintellect.ai/faq

AWS:

- EC2 Spot pricing: https://aws.amazon.com/ec2/spot/pricing/
- EC2 P4 instances: https://aws.amazon.com/ec2/instance-types/p4/
- EC2 P5 instances: https://aws.amazon.com/ec2/instance-types/p5/
- EC2 P6 instances: https://aws.amazon.com/ec2/instance-types/p6/
- EC2 P6-B200 launch note: https://aws.amazon.com/blogs/aws/new-amazon-ec2-p6-b200-instances-powered-by-nvidia-blackwell-gpus-to-accelerate-ai-innovations/
- EC2 accelerated instance specs: https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html

## Hardware Profiles

Repo hardware profile names:

```text
a100_80gb_sxm
a100_80gb_pcie
h100_80gb_sxm
b200
```

Use these profile names consistently in `local_benchmark.py`,
`autotune.sweep`, NCU scripts, result filenames, and analysis tables.

Useful cloud targets:

```text
A100:
  Prime: A100_80GB offers, SXM preferred if price is reasonable
  AWS:   p4d.24xlarge for 8x A100 40GB
  AWS:   p4de.24xlarge for 8x A100 80GB, where available

H100:
  Prime: H100_80GB offers
  AWS:   p5.4xlarge for 1x H100, where available
  AWS:   p5.48xlarge for 8x H100
  AWS:   p5e/p5en variants for H100/H200-class variants, region dependent

B200:
  Prime: B200 offers, where available
  AWS:   p6-b200.48xlarge for 8x B200, where available
```

If the provider exposes only 8-GPU instances, still run single-process
benchmarks. PyTorch will use one visible device unless code or environment
selects more.

## Prime Setup

Install Prime CLI:

```bash
uv tool install prime
```

Configure auth. Do not commit keys or echo them into shell history if avoidable:

```bash
prime config set-api-key
prime config set-ssh-key-path
prime config view
```

The API key needs at least:

```text
Availability -> Read
Instances -> Read and write
```

Prime docs also support `prime login`; use whichever matches the key/account
setup.

## Prime Capacity

Prime GPU type filters to try:

```text
A100_80GB
H100_80GB
B200
```

Provider naming can vary. Use `prime availability list --help` and inspect the
exact GPU type strings shown by Prime if a filter returns nothing.

List available offers:

```bash
prime availability list --gpu-type A100_80GB --gpu-count 1
prime availability list --gpu-type H100_80GB --gpu-count 1
prime availability list --gpu-type B200 --gpu-count 1
```

Useful filters:

```bash
prime availability list --gpu-type A100_80GB --gpu-count 1 --regions united_states,canada
prime availability list --gpu-type H100_80GB --gpu-count 1 --regions united_states,canada
prime availability list --gpu-type A100_80GB --gpu-count 1 --socket SXM
prime availability list --gpu-type A100_80GB --gpu-count 1 --socket PCIe
```

Prefer:

```text
gpu_count: 1
socket: SXM if price is reasonable, PCIe if much cheaper
image: CUDA/PyTorch image if available
pricing: spot/low-cost offer when available
disk: at least 100GB
```

The public CLI docs list availability filters but do not show a universal
`--spot` flag. If the availability table exposes an `isSpot`/spot-priced offer,
pick that offer. If not, use the cheapest acceptable offer and treat it as
on-demand fallback.

## Prime Launch

Interactive creation is safest because provider-specific options differ:

```bash
prime pods create
```

When prompted, choose:

```text
GPU: A100_80GB, H100_80GB, or B200
GPU count: 1
spot/low-cost offer: yes if offered
image: CUDA 12 + PyTorch image if available
disk: 100GB or more
name: qr-<gpu>-sweep
```

Non-interactive creation can use an availability `--id` or provider `--cloud-id`
from `prime availability list`:

```bash
prime pods create \
  --id <availability-id> \
  --name qr-a100-sweep \
  --disk-size 120
```

If using provider IDs directly:

```bash
prime pods create \
  --cloud-id <cloud-id> \
  --gpu-type A100_80GB \
  --gpu-count 1 \
  --name qr-a100-sweep \
  --disk-size 120
```

Use `prime pods create --help` on the machine because CLI flags may change.

Monitor and SSH:

```bash
prime pods list
prime pods status <pod-id>
prime pods ssh <pod-id>
```

Terminate immediately when done:

```bash
prime pods terminate <pod-id>
prime pods list
```

Spot instances can be interrupted. Keep outputs under `results/` and copy them
back frequently.

## AWS Setup

Install/configure AWS CLI:

```bash
brew install awscli
aws configure
aws sts get-caller-identity
```

Required setup:

```text
EC2 key pair
security group allowing SSH from your IP
VPC/subnet in target region
EC2 GPU service quota for the instance family
budget/cost alert
```

Do not commit AWS credentials. Prefer a named AWS profile:

```bash
export AWS_PROFILE=qr-gpu
export AWS_REGION=us-west-2
```

## AWS Capacity

Check spot price history:

```bash
aws ec2 describe-spot-price-history \
  --instance-types p4de.24xlarge p5.4xlarge p5.48xlarge p6-b200.48xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'SpotPriceHistory[*].[InstanceType,AvailabilityZone,SpotPrice,Timestamp]' \
  --output table
```

Check instance type availability in a region:

```bash
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=p4de.24xlarge,p5.4xlarge,p5.48xlarge,p6-b200.48xlarge \
  --query 'InstanceTypeOfferings[*].[InstanceType,Location]' \
  --output table
```

High-end GPU Spot capacity can be scarce. If Spot fails, try another AZ/region,
use smaller single-GPU P5 where available, or fall back to Prime.

## AWS AMI

Prefer an AWS Deep Learning AMI with CUDA/PyTorch already installed.

Find recent Ubuntu DLAMI images:

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters 'Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu*' \
  --query 'Images | sort_by(@, &CreationDate)[-5:].[ImageId,Name,CreationDate]' \
  --output table
```

Set:

```bash
export AMI_ID=<ami-id>
export KEY_NAME=<ec2-keypair-name>
export SECURITY_GROUP_ID=<sg-id>
export SUBNET_ID=<subnet-id>
```

## AWS Launch

Example H100 single-GPU P5:

```bash
aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type p5.4xlarge \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=qr-h100-spot}]'
```

Example A100 8-GPU:

```bash
aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type p4de.24xlarge \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=qr-a100-spot}]'
```

Example B200:

```bash
aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type p6-b200.48xlarge \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --subnet-id "$SUBNET_ID" \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=qr-b200-spot}]'
```

Capture instance id:

```bash
export INSTANCE_ID=<instance-id>
aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].[State.Name,PublicDnsName,PublicIpAddress]' \
  --output table
```

SSH:

```bash
export HOST=<public-dns-or-ip>
ssh -i ~/.ssh/<key>.pem ubuntu@"$HOST"
```

Terminate when done:

```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' \
  --output text
```

Also remove orphaned EBS volumes if `DeleteOnTermination` was not set.

## Remote Bootstrap

On the rented machine:

```bash
nvidia-smi
git clone https://github.com/justinchiu/qr-comp.git
cd qr-comp

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --group practice
```

Verify CUDA PyTorch:

```bash
uv run --group practice python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
PY
```

If the image already has a working CUDA PyTorch environment, keep using it. If
`uv` installs a CPU-only torch wheel (`torch.cuda.is_available()` is `False`),
force the CUDA wheel index to match the image's CUDA toolkit, e.g. for CUDA 12.x:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Match the `cuNNN` suffix to the driver/toolkit reported by `nvidia-smi`
(`cu121`, `cu124`, `cu128`, ...). Re-run the verify snippet above and confirm
`torch.version.cuda` is non-empty before benchmarking. If the image's own Python
already has working CUDA torch, just use it instead of fighting `uv`.

## Secrets on the Remote Machine

Do not `scp` credential files (`~/.aws/credentials`, `~/.config/prime/`,
`*.pem`, `.env`) to the rented box. Prefer short-lived, scoped access:

- Clone over HTTPS for public repos, or use SSH agent forwarding
  (`ssh -A`) for private ones — do not copy private keys onto the instance.
- For Prime/AWS calls made *from* the remote box, export keys into the shell
  for that session only; do not write them to disk or shell history
  (prefix the command with a space, or `unset` after use).
- Terminate the instance when done — anything left on a spot box is gone, but
  anything written to its EBS/disk before that is recoverable until it is.

## Runbook: A100

Use A100 for cheap CUDA correctness and sweep iteration.

Correctness:

```bash
uv run --group practice python local_benchmark.py \
  --hardware a100_80gb_sxm \
  --suite official \
  --mode test
```

For PCIe A100:

```bash
uv run --group practice python local_benchmark.py \
  --hardware a100_80gb_pcie \
  --suite official \
  --mode test
```

Sweep:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware a100_80gb_sxm \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes auto \
  --repeats 5 \
  --output results/a100_512_1024_sweep.csv
```

NCU baseline profile if Nsight Compute is available:

```bash
QR_HARDWARE=a100_80gb_sxm \
QR_MODULE=baselines.geqrf_baseline \
QR_CASE_INDEX=3 \
NCU_SET=roofline \
./scripts/ncu_qr.sh
```

## Runbook: H100

Use H100 for a closer high-end CUDA profile than A100. H100 is still not B200,
but its Hopper Tensor Core and memory behavior is more relevant than A100.

Correctness:

```bash
uv run --group practice python local_benchmark.py \
  --hardware h100_80gb_sxm \
  --suite official \
  --mode test
```

Sweep:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware h100_80gb_sxm \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes auto \
  --repeats 5 \
  --output results/h100_512_1024_sweep.csv
```

Optional NCU:

```bash
QR_HARDWARE=h100_80gb_sxm \
QR_MODULE=baselines.geqrf_baseline \
QR_CASE_INDEX=3 \
NCU_SET=roofline \
./scripts/ncu_qr.sh
```

## Runbook: B200

Use B200 for final profiling and dispatch decisions.

Correctness:

```bash
uv run --group practice python local_benchmark.py \
  --hardware b200 \
  --suite official \
  --mode test
```

Profile the stable `torch.geqrf` baseline first:

```bash
./scripts/profile_geqrf_baseline.sh
```

Sweep:

```bash
uv run --group practice python -m autotune.sweep \
  --hardware b200 \
  --n 512,1024 \
  --batch 16 \
  --cases dense,rankdef,nearrank,clustered,rowscale,nearcollinear,mixed \
  --variants python_geqrf,python_blocked,cholesky_probe \
  --block-sizes auto \
  --repeats 5 \
  --output results/b200_512_1024_sweep.csv
```

Profile important cases:

```bash
GEQRF_BASELINE_CASES="3 7 9 10 4 8 11" ./scripts/profile_geqrf_baseline.sh
QR_HARDWARE=b200 QR_CASE_INDEX=3 NCU_SET=roofline ./scripts/ncu_qr.sh
```

## NCU Workflow

NCU is a first-class part of the workflow. Do not promote a kernel from wall
time alone when the bottleneck is unclear.

For the `torch.geqrf` baseline:

```bash
mkdir -p logs
GEQRF_BASELINE_CASES="3" ./scripts/profile_geqrf_baseline.sh 2>&1 | tee logs/geqrf_case3.log
tar -czf qr-geqrf-case3-$(hostname)-$(date +%Y%m%d-%H%M%S).tgz profiles logs results
```

For a custom module:

```bash
mkdir -p logs
QR_HARDWARE=b200 \
QR_MODULE=submission \
QR_CASE_INDEX=3 \
NCU_SET=roofline \
./scripts/ncu_qr.sh 2>&1 | tee logs/submission_case3.log
tar -czf qr-submission-case3-$(hostname)-$(date +%Y%m%d-%H%M%S).tgz profiles logs results
```

Useful overrides:

```bash
QR_CASE_INDEX=7 NCU_SET=roofline ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_KERNEL_NAME='.*qr.*' ./scripts/ncu_qr.sh
QR_CASE_INDEX=3 NCU_LAUNCH_SKIP=10 NCU_LAUNCH_COUNT=1 ./scripts/ncu_qr.sh
```

Do not act on a profile until the same run also passed `local_eval` or
`local_benchmark.py --mode test`.

## Bring Results Back

Prime:

```bash
# Use the SSH details printed by Prime for scp.
scp <prime-scp-host>:~/qr-comp/qr-*.tgz .
```

AWS:

```bash
scp -i ~/.ssh/<key>.pem ubuntu@"$HOST":~/qr-comp/qr-*.tgz .
```

Unpack locally outside git or into ignored directories:

```bash
mkdir -p remote-artifacts
tar -xzf qr-*.tgz -C remote-artifacts
```

Open NCU reports with `ncu-ui` if installed:

```bash
ncu-ui remote-artifacts/profiles/**/*.ncu-rep
```

If shell globstar is unavailable, open the specific `.ncu-rep` files under
`remote-artifacts/profiles/`. If `ncu-ui` is unavailable locally, keep the
`.ncu-rep` files and summarize the key metrics from the remote `ncu` console
output.

Minimum handoff:

```text
1. .ncu-rep report
2. logs/*.log console output
3. results/*.csv sweep/timing output
```

Fill this table for each profiled case:

```text
gpu | provider | instance | case | module | mean ms | dominant kernel | bottleneck | evidence | next action
A100| prime    | ...      | 3    | geqrf  | ...     | ...             | ...        | ...      | ...
A100| aws      | p4de     | 3    | geqrf  | ...     | ...             | ...        | ...      | ...
H100| aws      | p5       | 3    | geqrf  | ...     | ...             | ...        | ...      | ...
B200| aws      | p6-b200  | 3    | geqrf  | ...     | ...             | ...        | ...      | ...
```

Minimum evidence to record:

```text
DRAM throughput and bytes
L2 hit rate
SM active / SM throughput
achieved occupancy
registers per thread
shared memory per CTA
top warp stall reasons
number of launched kernels
kernel duration for dominant kernels
```

Interpretation:

```text
memory-bound:
  high DRAM, low SM, arithmetic intensity below ridge

compute-bound:
  high SM/tensor utilization, DRAM not saturated

latency/sync-bound:
  low DRAM and low SM, stalls dominate

launch-bound:
  many short kernels and high total launch overhead

occupancy-limited:
  low active warps due to registers/shared memory
```

## What Each GPU Can Decide

A100/H100 results can answer:

- does the CUDA path run at all?
- does the QR checker pass on real CUDA?
- are launch counts excessive?
- do block-size trends look plausible?
- does Cholesky fail hard cases as expected?

Only B200 results should decide:

- final block size
- final roofline classification
- final dispatch choice
- Popcorn leaderboard strategy

## Shutdown Discipline

Spot instances can be interrupted. Use `tmux`, tee logs, and package artifacts
often. Do not leave expensive GPU instances running.

Before shutdown:

```bash
tar -czf qr-final-results-$(hostname)-$(date +%Y%m%d-%H%M%S).tgz \
  results \
  profiles \
  logs
```

Then copy the archive back and terminate the Prime pod or AWS EC2 instance.
