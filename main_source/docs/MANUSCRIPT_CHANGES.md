# Relationship to the user's current manuscript

Preserved: READER title; observed-prefix versus information availability question;
BiTE readout/fusion attribution; causal temporal-spatial tokenizer; two depthwise
TCNs; static featurewise gate; endpoint-only research objective; bounded trial
scope; within-subject three-dataset evaluation; subject-level aggregation.

Made explicit: sample/token indexing; actual partial-window denominator; residual
initialization; projection biases and position table; train-only coordinate
standardization; crop-relative times; independent minibatch RNG; finite causal
checks; actual STFT bin rule; fixed hardware timing scope.

New measured evidence after the run: all5endpointmodels, all4coreprefixcurves,
FF andCompact--Meantrainedcontrols,trained/nondegeneratecausality,pairedCIs,
normalizedaccuracy-durationarea,parametercounts andtiming,diagnostics.

Removed until separately justified: old numerical means, rounded-up specialist
table, old one-seed LOSO table, inherited HGD result. New tables must replace the
old ones together, not selectively replace only favorable model values.

Related work is shortened to the relevant established sources. The manuscript
uses the existing supplied figure unchanged. Optional additional citations and
expanded related work can be restored after the result-based draft is reviewed;
no literature novelty guarantee is made by code generation.

The pipeline fills Abstract and Results with actual new averages and intervals,
adds measured-control interpretations and a data-dependent conclusion, and
creates all LaTeX table/figure inputs. It does not fabricate authors, certify the
rawdata,claimpreregistration,or claim statistical superiority on unmeasured data.

The template cannot compile as a research paper before actual result inputs
exist. Tests may compile an explicitly watermarked SYNTHETIC LAYOUT TEST in a
private temporary directory; those fixture tables are never included in this ZIP
or accepted as real-study outputs.
