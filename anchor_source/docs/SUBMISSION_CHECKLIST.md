# Final author review before submission

The computation can establish a measured complete cohort, not authorize submission.
Review `cohort_status.json` and the following matters.

1. All420newruns andcompletepairedpredictions: no legacy rows silently mixed;
   failures retained and resolved scientifically,not byremovingbadmodelcells.
2. Trained/nondegeneratecausality errors andpositivecontrols;readtheactualmaxima.
3. Prepared-data provenance:rawrecords,trial/blocksplit,channelorder,cropreference,
   SSVEPindexrounding,original artifact handling and no invalid fitrole. Existing
   row IDs/hashes alone cannot certify the entire raw pipeline.
4. Accurate history of test exposure. This follow-up is not an unseen benchmark,
   and source/recipe choices were made after viewing earlier outcomes.
5. Author names,affiliations,correspondence,approvals,acknowledgment/funding and any
   relevant institutional/public-data ethicsstatement. Placeholderauthors block
   finalsubmission;thepipeline willnot invent these details.
6. Supportedinterpretation ofpositive,nullandnegativecomparisons. Strongerclaims
   are not obtained automatically by bootstrap CIs or finishingthe scripts.
7. References:confirm publishedversions,DOIs,years and codecommit in actualused
   baseline. No exhaustive bibliographic audit is claimed in this package.
8. Officialworkshop page limit,anonymity,format,supplementpolicy,figures/readability,
   PDFfonts andvisualinspection. Sourcecompilation isnot a page-limit certification.
9. Data/code licensingandprivacy beforeGitHubrelease:upstreamBiTE is fetched not
   redistributed;rawarrays,weightsandprivateconfigs are excluded fromcodeupload.

Add approved author LaTeX to `submission_review.json` before the collector runs,
without editing `study.json` or frozen source. If adding it later, regenerate the
paper in a NEW review directory from existing analysis; preserve the earlierdraft.
The finalanswer needs the actualreturnarchive to audit finalfiguresandwording.
