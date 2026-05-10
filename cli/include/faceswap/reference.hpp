#pragma once

#include "faceswap/face_analyser.hpp"
#include "faceswap/types.hpp"

#include <vector>

namespace faceswap {

/// Single-pass video scan that fills in spec.ref_emb / ref_frame / ref_votes
/// / ref_pool for every source. Greedy per-gender cluster assignment so two
/// same-gender sources land on different recurring people.
///
/// Same algorithm as worker.py::extract_reference_embeddings.
void extract_reference_embeddings(
    const fs::path& video_path,
    const FaceAnalyser& analyser,
    std::vector<SourceSpec>& sources,
    int min_face_w = 25,
    int max_samples = 120);

}  // namespace faceswap
