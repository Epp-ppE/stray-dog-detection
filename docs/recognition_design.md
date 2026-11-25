# Dog recognition (re-identification): design

**Goal.** Decide whether a dog seen in one clip is the *same individual* as a dog seen in
another clip, at another time or at another camera. Detection says "a dog is here".
Recognition says "this is dog #12, last seen yesterday at 08:05 at Camera3". That is what
turns per-clip counts into a population estimate, per-dog visit histories and a list of dogs
still to vaccinate or sterilise.

This document is the design the assignment asks for. A small working demo built on it is in
`src/reid_demo.py` (results in `outputs/reid/`).

---

## 1. Where recognition sits in the pipeline

```
clip -> frames (3 fps) -> detector -> ByteTrack -> confirmed tracks
                                                     |
                                     best crops per track (sharp, large, unoccluded)
                                                     |
                                     embedding model -> one vector per track
                                                     |
                     match against the gallery of known dogs (+ time/place prior)
                                                     |
                         same dog -> add to its history | new dog -> new gallery entry
```

The unit of recognition is a **track**, not a single frame. A track already groups several
views of one dog within a clip, so it gives several crops to average, which is more robust
than one picture. Within-clip identity is handled by the tracker. Recognition only has to
link tracks *across* clips.

## 2. Data collection

| Step | What | Why |
|---|---|---|
| Free positives | Every confirmed track = several crops of one dog | Same-dog pairs with no labelling at all |
| Free negatives | Two tracks visible **in the same frame** are different dogs | Hard negatives (same place, same light) with no labelling |
| Human linking | A person groups tracks across clips into named dogs (dog-01, dog-02, ...), using a page like the box-labelling tool that shows track crops side by side | The only manual step; about 20-30 dogs give a usable test set |
| Distinctive marks | For each named dog, note coat colour, patches, ear shape, collar or ear-notch (sterilised dogs in Thailand are often ear-notched) | Helps labellers stay consistent and gives attributes for search |
| Keep collecting | Run the cameras over weeks, not 3 days; add a camera that sees dogs side-on at eye level | Most crops here are top-down while eating, with the head in the bowl |

**Splits for evaluation:** by *dog* and by *day*. Test dogs must not appear in training,
and the gallery and query images should come from different days, so the model can't win
by matching the same lighting or the same moment.

## 3. Features and models

1. **Baseline (no training):** a general pretrained image model used as a feature
   extractor. The demo uses the YOLO11 ImageNet classifier's pooled features. Other options are
   DINOv2 or CLIP image features. They capture coat colour, patches and body shape.
   Cosine similarity compares two tracks.
2. **Fine-tuned re-identification model:** start from the same backbone, add an embedding
   head and train with a metric-learning loss (triplet loss with hard-negative mining, or
   ArcFace / sub-center ArcFace), as in person and animal re-ID. Train on public animal
   re-ID data first (dog-face and pet re-ID datasets, plus wildlife re-ID collections), then
   fine-tune on the locally linked tracks.
3. **Part-aware features:** dogs here are often partly hidden by the fence, bowls or each
   other. Pooling features from body parts (head, torso, legs) separately, or masking
   hidden regions, keeps one visible patch from dominating.
4. **Track-level aggregation:** embed the best 5-10 crops of each track (largest, sharpest,
   detector-confident, not cut off at the edge) and average them. Hard cases can use the
   minimum distance over crop pairs instead.
5. **Separate day and night galleries:** infrared removes colour, so a brown dog and a
   black dog can look identical. Night tracks are matched with shape-only features and a
   higher threshold, or linked to day tracks only through time and place.

## 4. Matching

* **Similarity:** cosine similarity between track embeddings.
* **Time and place prior:** two tracks on the same camera less than about 2 minutes apart
  are very likely the same dog (a motion-triggered burst), so the similarity is raised.
  Dogs can't be in two places at once, so overlapping tracks in the same frame are forced
  to be different dogs.
* **Open-set decision:** if the best match is below a threshold, the track is a **new
  dog** and starts a new gallery entry. The threshold is chosen on held-out dogs to balance
  false merges against false splits.
* **Clustering view:** for population counting, all tracks are clustered (agglomerative,
  average linkage) with the same-frame "cannot link" constraints, and the number of
  clusters estimates the number of distinct dogs.
* **Human in the loop:** low-confidence links are queued for a person to confirm. Each
  confirmed link becomes new training data.

## 5. Evaluation

| Metric | Question |
|---|---|
| Rank-1 / Rank-5 accuracy | Is the correct dog the top (or a top-5) gallery match? |
| mAP (re-ID) | How well does the system rank all images of the right dog? |
| Pairwise precision / recall | Of the "same dog" links made, how many are right, and how many true links are found? |
| Count error | Does the number of clusters match the number of distinct dogs? |
| Per condition | Day vs night, same camera vs cross-camera, puppy vs adult |

## 6. Challenges seen in this footage

* **Look-alike dogs:** several black puppies at Camera1 and several white-and-tan puppies
  at Camera3 differ only in small patches.
* **Night infrared:** colour disappears and glare washes out texture.
* **Viewpoint:** mostly top-down and from behind while eating, so faces (the most
  distinctive part) are rarely visible.
* **Occlusion:** chain-link fence over the whole Camera1 view, plants at Camera4, and dogs
  overlapping each other at the bowl.
* **Puppies grow:** appearance changes within weeks, so the gallery must be refreshed.
* **Lens distortion:** wide-angle cameras stretch dogs near the edge.
* **Low resolution for far dogs:** dogs on the road at Camera3 are only a few dozen pixels tall.
* **Short clips (3-30 s):** some tracks have only a few usable crops.
* **Collars and ownership:** a collar suggests an owned dog. Telling stray from owned needs
  extra cues and local knowledge, and is beyond appearance alone.

## 7. Privacy and ethics

People appear in the footage (feeders, a person on a motorbike, a hand on the lens). The
recognition system must only build galleries of **dogs**. Person detections should be used
only to *discard* frames or blur people, never to identify them.
