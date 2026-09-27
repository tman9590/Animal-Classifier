# Model card: North Carolina Species Detector

## Intended behavior

The deployable model is a single-stage object detector for Scrypted NVR. Each
box receives one of 529 North Carolina wildlife and poultry labels or
`unknown`. Deployed labels use species-level common names. Scientific names are
retained in the catalog and teacher prompts but are not shown in Scrypted.

## Teacher and student

Pseudo-labels are produced by MegaDetector V6 compact and BioCLIP 2. For camera
clips, detections are associated by bounding-box overlap and BioCLIP
probabilities are smoothed with an exponential moving average. A track becomes
`unknown` below the configured confidence threshold. These annotations train a
YOLO11 Small student that is exported to the tensor layouts consumed by
Scrypted's existing custom-object-detection parser.

The deployed student does not execute BioCLIP or query iNaturalist. Its class
list is fixed at training time. Rebuilding the catalog and retraining are
required to change regions or add species.

## Geographic and taxonomic scope

The vocabulary is derived from research-grade observations within iNaturalist
place 30, North Carolina, USA. It contains terrestrial vertebrates in
Mammalia, Aves, Reptilia, and Amphibia with at least 20 observations, plus an
explicit poultry list. Marine fish, invertebrates, and taxa with sparse North
Carolina evidence are outside the default scope.

Presence in the list means that a taxon has been observed in North Carolina. It
does not mean that the species is plausible at every address, habitat, season,
or time of day in the state.

## Confidence and unknown

The BioCLIP threshold controls pseudo-label creation, not a calibrated
probability that a species is present. The student learns `unknown` from animal
boxes that the teacher cannot classify confidently. Its runtime score is also
not a calibrated biological probability.

## Limitations

- Species with similar appearance, hybrids, juveniles, domestic breeds, and
  partial views are likely confusion pairs.
- Night vision, small subjects, motion blur, rain, backlighting, and animals at
  the frame edge reduce accuracy.
- iNaturalist and BioCLIP training data are long-tailed. Frequently photographed
  birds and mammals have more evidence than secretive species.
- A class list with hundreds of species increases fine-grained coverage but
  requires substantial examples per class. Sparse classes should remain
  experimental until validated on held-out footage.
- IoU association can switch identities when animals cross. Smoothing may also
  delay a correct label after an early error.
- The model is for notification and search assistance. It is not evidence for
  a scientific record, wildlife-management decision, or safety-critical action.

## Required validation

Measure per-class precision and recall on clips excluded from training. Report
macro averages so common species do not hide failures on rare ones. Maintain a
confusion matrix, unknown recall, day/night slices, and separate results for
wild birds, poultry, mammals, reptiles, and amphibians.
