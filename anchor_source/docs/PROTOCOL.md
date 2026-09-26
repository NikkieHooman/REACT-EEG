# Frozen prospective follow-up specification

This is a follow-up chosen after seeing the old study's results, not retrospective
preregistration. It is a newly implemented manuscript formulation. New parameters,
initialization, data order and constraint timing must not be represented as proven
historical implementation details.

## Models and exact differences

The four core models share tokenizer, first TCN, positions and classifier, all
constructed in the same order from seed 2025, 2026 or 2027. The extra reader is
constructed from an isolated CPU stream at seed 918273. Shared initial state hashes
and independent minibatch-order hashes are compared within subject/seed. Identical
dropout masks across different architectures are not claimed.

Tokenizer: three zero-left-padded, stride-one, bias-free Conv2d branches of 16
maps each and temporal widths 16/32/64; BatchNorm with 48 features; grouped spatial Conv2d
48-to-96 with groups48 and width equal to channel count; BatchNorm with 96 features; LeakyReLU with slope 0.2.
No radial/envelope path, analytic front end, content gate or per-subject test-time
adaptation is included. Pool size32 for MI,4 for SSVEP. Nonoverlapping sum/count
pooling uses only real samples in a final partial token; it does not overlap a
full-width window or count right-padded zeros. Dropout 0.3, biased Linear 96-to-64,
and learned position vectors N(0,.02). The table contains 32 MI or64 SSVEP positions.

Reader: three residual blocks, two bias-free depthwise Conv1d per block, kernel 6,
dilations 1/2/4. Each convolution is followed by affine BatchNorm, LeakyReLU with slope 0.2,
and dropout 0.3. BatchNorm epsilon 1e-5, momentum 0.1, running statistics enabled. First
convolution uses Xavier uniform; second uses zero weights. BatchNorm starts with
unit scale, zero bias/running mean,unit running variance. Residual addition has no
subsequent activation, giving identity initialization. Evaluation disables dropout
and fixes BN statistics. The receptive field is71 tokens. This is bounded-trial,
not unlimited streaming inference.

Compact reads the final forward state. Compact--Mean reads the equal-token mean
of all forward states, including the partial last token. READER reads original
and reversed positioned tokens, combining last states with a learned 64-feature
sigmoid gate initialized at 0.5. FF supplies original-order tokens to both
independently parameterized readers. FF changes order and resulting anchor
jointly, so it is not a perfect isolation of direction from anchor.

The biased classifier is64-to-K. Spatial-filter row norms are projected to <=1,
classifier-row norms to <=.25, once before training and after each optimizer
step. Core models do not modify weights during inference. Core parameter counts:

| Dataset | Compact / Mean | READER / FF |
|---|---:|---:|
| 2A | 15,780 | 18,916 |
| 2B | 13,826 | 16,962 |
| SSVEP | 17,004 | 20,140 |

The increment is 3,136. These are NEW instantiated counts, not the native legacy
17,828/21,316 counts. The position-table allocation is an explicit new choice.

BiTE uses unchanged pinned upstream source, imported separately. The adapter and
STFT follow the supplied BIBM_V2 adapter calling convention: F1=16,D=4,kernel 64,
pool 32, nominal4--40 Hz for MI;kernel 32,pool 8,8--64 Hz for SSVEP. STFT uses Hann window,
FFT/window length k, hop1,centerTrue,constantpadding,onesidedmagnitude; frequency
indices floor(low*k/fs) through floor(high*k/fs),inclusive. Report this actual bin
rule rather than implying exact band-edge frequencies. Input is B,C,T, spectrum
B,F,frames,C; logits are the first upstream return element. Endpoint only is
assumed; unsupported BiTE prefix lengths are not guessed. BiTE's native constraints
and internal modules are unchanged. Source hashes are verified every task.

## Data and fitting roles

The declared files use NPZ data[B,C,T], label[B] zero-based integers. 2A uses T/E
sessions. 2B concatenates01T/02T/03T for training and04E/05E for test. SSVEP preserves
its existing train/test files. MI preparation can store 1001 samples: loader keeps
the first 1000 explicitly. SSVEP must contain 256 samples. No resampling/new reference/filter
or new train/test split is performed.

Load training arrays first. Fit a separate scikit-learn StandardScaler per
subject on Xtrain.reshape(N,-1),float64; retain mean/scale/variance indexed by
channel and within-trial sample. Cast normalized data tofloat32. Prefix input
selects the corresponding scaler coordinates. No test-fitted normalization,
whitening or trial-specific normalization is used. Exact duplicates are rejected
across prepared train/test rows; lack of duplicates is not proof of original
block independence. File and array checksums and file+row IDs are saved. Those
IDs are not falsely labelled original raw trial IDs.

Each subject/seed task finishes all five model trainings and saves final
checkpoints before opening its held data. The preflight reads training data
only. File existence checks at setup do not read held arrays.

The supplied legacy preparation source uses session-level channel minimum-value
replacement for MI and, for SSVEP, start=int(.135*256)=34 before taking256samples.
Whether every existing NPZ was actually created by that version is not established
by its filename/shape. Preserve this as an author provenance check. Session-wide
raw cleaning need not be causal within a stream. The fresh study claims only a
prepared-input model boundary. Observation times are relative to the prepared
crop, not necessarily stimulus/cue onset or acquisition latency.

## Training

All 420 models use600 epochs,Adam lr.002,weight_decay.002,betas(.9,.999),eps1e-8,
batch64,labelsmoothing.1,L2globalgradient clipping5,cosineLRtozero with epoch-end
scheduler step. Incomplete last minibatches are retained. No augmentation,
validation-based selection, prefix supervision, early stopping or per-epoch held
accuracy monitoring. Model construction seed=k; training RNG seed=k+10000.
Minibatch generator is separateCPU seed=k+20000+epoch. These explicit conventions
need not match historical samplers. FP32 withTF32disabled,deterministicalgorithms
required. A backend error is not silently changed to nondeterministic execution.

Every 25 epochs the resumable checkpoint includes optimizer,scheduler,random states,
recipe/source/data signatures and history. Final weights are round-tripped and
compared with saved training-input logits before held evaluation. A fresh fit
is not evidence that old results reproduced.

## Outcomes

All four core models are evaluated on one checkpoint each at identical inputs.
MI grid: 256..992 by 32 plus 250, 500, 750, 1000 (28 points, 1--4 s). SSVEP grid: 64..256 by 4
(49 points, 0.25--1 s). No seed ensembling: accuracy first, then seeds within subject.
Each prediction record includes logits, labels, IDs and exact sample counts.

Primary comparison: 2A READER-minus-Compact normalized trapezoidal area over 1--4 s.
Secondary: endpoint and curve area versus FF/Mean and other datasets. Accuracy and
curve area are summarized per subject. Paired percentile intervals use 10,000
subject bootstrap samples, seed 170926, retaining the same sampled subject indices
throughout each paired curve. These intervals are conditional on the observed
trained models; they are pointwise rather than simultaneous, unadjusted descriptive intervals.
No significance stars or p-values are generated by default. Display all fixed
contrasts,including null or negative differences. Do not change intervals/grids
because a result is unfavorable.

## Causality and interpretation

Test every trained core model and a nondegenerate in-memory temporal-weight copy on
4 label-independent prepared examples and4 synthetic inputs. MI boundaries:
32,64,128,250,256,500,512,750,768,992,1000. SSVEP boundaries:
4,8,64,65,127,128,192,256. Compare independent full-input sample-feature routing
with direct truncated inference; check repeated calls, batch composition, unchanged
state and future suffix replacement. Trace second-reader token counts. Usea predeclared 1e-4
absolute error. The endpoint future test is not applicable. A deliberately injected
unobserved-sample dependency must be detected. Record errors and failed checks.
Nondegenerate copies randomize only TCN Conv1d and reset TCN BatchNorm; original files stay unchanged.

READER diagnostics report gate values, branch magnitudes, weighted magnitudes, cosine,
context perturbations and post-hoc shared-head branch accuracies. These are not trained
baselines or causal explanations of physiological mechanisms.

## Timing

Subject 1, seed 2025 per dataset, all 5 models, batch size 1, 10 warm-ups and 50 synchronized repetitions.
Measure first/last requested prefixes; core models also use naive and reused-feature curves.
BiTE STFT is included. Normalization, file I/O, CPU/GPU transfer and acquisition are excluded. Devices
recorded per run; do not pool different hardware. Frozen statistics and all within-dataset models
for thatcell are timed on the same allocation. No latency or accuracy win is guaranteed.

## Exclusions and publication

Legacy HGD, LOSO and deadline specialists are excluded rather than relabelled. The
paper remains focused on within-subject endpoint-trained prefix representations.
The collector requires all 420 model results and all core causality checks, verifies
predictions and writes actual numbers. It does not infer authors or raw data provenance.
A complete compiled paper still requires author, reference, venue and visual review.
