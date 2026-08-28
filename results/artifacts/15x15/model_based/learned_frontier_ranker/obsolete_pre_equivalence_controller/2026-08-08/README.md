# Obsolete pre-equivalence V12 controller artifacts

These smoke artifacts used an experimental V12 receding-horizon/resource mode
that did not reproduce the stronger V11 planner when no neural checkpoint was
loaded.  They are retained only for auditability and must not be used for
training, checkpoint selection, or reported results.

The current V12 runner now follows V11's two-phase execution and replanning
schedule exactly when no checkpoint is supplied.  Its only learned component
is the uncertainty-gated phase-1 target ranker.
