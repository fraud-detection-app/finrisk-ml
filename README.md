# finrisk-ml

The **model** pipeline of the financial risk assistant: train a fraud model in SageMaker, check it with a
gate, register it, let a person approve it, and move the approved version through `dev -> sit -> pre-prod ->
prod`. One of four repos (`finrisk-infra`, `finrisk-app`, **`finrisk-ml`**, `finrisk-knowledge`).

What runs in an environment is **three things**, each released by its own pipeline:

| thing | repo | how it is named | how the running app finds it |
| --- | --- | --- | --- |
| the app (code) | `finrisk-app` | image digest `sha256:...` | the service itself |
| the **model** | **this repo** | a **number**: model version 3 in the Model Registry | SSM `/finrisk/<env>/model` |
| the knowledge index | `finrisk-knowledge` | a version `YYYYMMDD-HHMMSS-<sha7>` | SSM `/finrisk/<env>/index` |

The app never restarts for a new model. It re-reads its pointer every minute and swaps the new model in.
That is why a model can be released (and rolled back) without building an image.

## The flow

```
 train.yml (environment `ml`)                       approve-model.yml           promote-model.yml
 ------------------------------------------         (environment                (one job per environment)
 image -> prepare-data -> train -> GATE -> register  `model-approval`)
                                         |                  |                  dev -> sit -> pre-prod -> prod
                                         v                  v                              (reviewers on the last two)
                              version N: PendingManualApproval -> Approved
```

| step | what happens | where |
| --- | --- | --- |
| image | build `finrisk-train:sha-<this commit>` FROM the app image pinned in `train/base_image.txt`; skipped if the tag exists | GitHub runner, ECR |
| `prepare-data` | `docker run` the training image: the app's own `scripts/run_pipeline.py --demo-rows 5000` makes the gold feature table; upload to `s3://<ml bucket>/data/<YYYYMMDD-HHMMSS>/train.parquet` | runner, S3 |
| `train` | a SageMaker training job `finrisk-train-<time>` on `ml.m5.large` (at most 30 min) runs the image; we print the status changes, billable seconds and final metrics | SageMaker |
| `evaluate` | **the gate** (below). Fails with exit 3: nothing is registered | runner |
| `register` | a numbered version in the group `finrisk-fraud-model`, `PendingManualApproval`, with the training image digest, the data URI, this repo's commit and the metrics as metadata | Model Registry |
| `approve` | a person approves or rejects (`approve-model.yml`, reviewer required) | Model Registry |
| `promote` | write `/finrisk/<env>/model`, then poll the app's `/ready` until `model.version` is the promoted number | SSM, the app |

### The gate

`config/train.yaml` holds the numbers. A candidate must pass **both** rules, using PR-AUC on the held-out test slice:

1. `PR-AUC >= 0.70` (the floor), and
2. `PR-AUC >= champion - 0.02` (the tolerance), where the **champion** is the model whose version is in
   `/finrisk/prod/model` (looked up in the registry). No champion yet: only the floor applies.

The first model only has to clear the floor; every later model must also not be noticeably worse than what runs in
production. Re-run `train` with `model_type = baseline` (a logistic regression) to watch the gate say no.

### Who may promote what

| environment | accepts |
| --- | --- |
| `dev`, `sit` | any version that is not Rejected (a person may want to look at it before approving it) |
| `pre-prod`, `prod` | **Approved** versions only (exit code 2 otherwise) |
| any | a **Rejected** version is never accepted |

Rolling back is the same command with the previous version number.

### The pointer

`promote` writes this JSON (Overwrite) to SSM `/finrisk/<env>/model`:

```json
{"version": 3, "model_data_url": "s3://<ml bucket>/models/finrisk-train-20261016-031500/output/model.tar.gz",
 "status": "Approved", "promoted_at": "2026-10-16T03:20:00+00:00", "run": "https://github.com/<org>/finrisk-ml/actions/runs/123"}
```

### The model package (`model.tar.gz`, flat)

| file | what |
| --- | --- |
| `model_v2.pkl` (`model_v1.pkl` for the baseline) | the model, its threshold and its feature list |
| `latest_model.joblib` | a pointer to the file above: `{"model_path": "model_v2.pkl"}` (a file name, never a path) |
| `features.parquet` | a copy of the gold table the model was trained on; the app scores `/api/predict/<row_id>` from it |
| `metrics.json` | `pr_auc`, `roc_auc`, `threshold`, `precision`, `recall`, `rows`, ... |

## Layout

```
VERSION                      0.1.0
config/train.yaml            hyperparameters, instance, gate thresholds (a change here starts a training run)
train/Dockerfile             the training image = the app image + the trainer
train/base_image.txt         the app image, pinned by digest (bumping it is a pull request)
train/sagemaker_train.py     runs INSIDE the image: reads the channel, trains with the app's code, writes the model package
pipeline/mlops.py            the CLI: prepare-data, train, evaluate, register, approve, promote, list, show
.github/workflows/           ml-ci.yml, train.yml, approve-model.yml, promote-model.yml
tests/                       pytest with a fake AWS (no network, no cost)
```

The training image starts FROM the app image on purpose: the model is trained with exactly the feature code and
the model code that will serve it. The trainer only decides *how the training is run*.

## Run it yourself

Everything below prints each AWS call before it makes it (account ids are shown as `<account-id>`).

```bash
pip install -r requirements.txt
python pipeline/mlops.py list                       # the versions in the registry (read-only)
python pipeline/mlops.py show --version 1
python pipeline/mlops.py evaluate --training-job finrisk-train-20261016-031500      # the gate, on an existing job
python pipeline/mlops.py promote --env dev --version 1 --no-wait                    # only moves the pointer
```

Names come from the environment, then from SSM (`/finrisk/shared/*`), then from the naming convention
(`ML_BUCKET`, `SAGEMAKER_ROLE_ARN`, `MODEL_PACKAGE_GROUP`, `AWS_REGION`). Exit codes: `0` ok, `1` failed,
`2` model version cannot be used here, `3` the gate said no.

The trainer also runs on a laptop, no AWS needed (the app's `src/` folder is next to this repo in the course workspace):

```bash
mkdir -p /tmp/in && cp <a gold transaction_features.parquet> /tmp/in/train.parquet
FINRISK_APP_SRC=../finrisk-app/src SM_CHANNEL_TRAIN=/tmp/in SM_MODEL_DIR=/tmp/m SM_OUTPUT_DATA_DIR=/tmp/o \
  MODEL_TYPE=baseline python train/sagemaker_train.py
```

## GitHub setup (once; `finrisk-infra/scripts/github_bootstrap.sh` does it)

The repo is **public**, so values that contain the AWS account id are **secrets**; everything else is a variable.
No AWS key exists anywhere: each job exchanges GitHub's OIDC token for one role, and the role's trust names the
exact Environment.

| name | kind | where | used by |
| --- | --- | --- | --- |
| `AWS_REGION` | variable | repository | all workflows (`ap-southeast-2`) |
| `AWS_ROLE_ARN` | secret | Environment `ml` | `train.yml` (role `finrisk-ml-train`) |
| `AWS_ROLE_ARN` | secret | Environment `model-approval` (reviewer) | `approve-model.yml` (role `finrisk-ml-approve`) |
| `AWS_ROLE_ARN` | secret | Environment `dev` | `promote-model.yml` job `promote-dev` (role `finrisk-ml-promote-dev`) |
| `AWS_ROLE_ARN` | secret | Environment `sit` | job `promote-sit` (role `finrisk-ml-promote-sit`) |
| `AWS_ROLE_ARN` | secret | Environment `pre-prod` (reviewer) | job `promote-pre-prod` (role `finrisk-ml-promote-pre-prod`) |
| `AWS_ROLE_ARN` | secret | Environment `prod` (reviewer) | job `promote-prod` (role `finrisk-ml-promote-prod`) |

The rule, the same in all four repos: a job **with** `environment:` reads the Environment secret `AWS_ROLE_ARN`
(each Environment holds its own role); a job without an environment reads a repo secret named after its role.
Every job in this repo has an Environment.

Every Environment accepts deployments from `main` only. `main` itself is protected: a pull request and a green
`test` check (`ml-ci.yml`) are required.

### Before the first training run

`train/base_image.txt` holds a placeholder digest. Copy the digest of the `finrisk-app` image that passed `dev`
(ECR -> `finrisk-app` -> the image -> "Copy digest"), put it on the `finrisk-app@sha256:...` line and open a pull
request. The workflow refuses to run with the placeholder, and says so.

## Tests

```bash
pip install -r requirements-dev.txt
FINRISK_APP_SRC=../finrisk-app/src python -m pytest      # the end-to-end training tests need the app's src
```

Without `FINRISK_APP_SRC` (and with no `../finrisk-app` next to this repo) only the end-to-end trainer tests skip.
