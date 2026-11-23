# hdmap-loc-utility

Maps are scored by AP or mIoU, but localization needs specific constraints:
lane lines fix lateral position and heading, while transverse elements
(crosswalks, stop lines) fix longitudinal position. This repository measures
how errors in predicted vectorized HD maps turn into pose errors, which element
classes make each pose component observable, and whether per-scene AP predicts
localization error.

**Status:** work in progress.
