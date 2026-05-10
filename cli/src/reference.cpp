// Reference embedding extractor. Same algorithm as the Python pipeline:
// 1. Sample up to N frames evenly across the video.
// 2. Detect every face in each sample, gated by min face width.
// 3. Bucket detections by gender. For each source, greedily find the
//    embedding cluster (within --ref-thresh cosine) with the most members
//    and lock that as the reference.
// 4. Same-gender sources mark their pool used so they don't both grab the
//    same recurring person.

#include "faceswap/reference.hpp"

#include <fmt/core.h>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace faceswap {

namespace {

float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return -1.0f;
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(s);  // both are pre-L2-normed
}

struct Candidate {
    int frame_idx;
    Face face;
};

}  // namespace

void extract_reference_embeddings(
    const fs::path& video_path,
    const FaceAnalyser& analyser,
    std::vector<SourceSpec>& sources,
    int min_face_w,
    int max_samples) {

    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened())
        throw std::runtime_error(fmt::format("could not open video: {}", video_path.string()));
    const int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (total <= 0)
        throw std::runtime_error("video has no frames or unknown duration");

    const int step = std::max(1, total / std::max(1, max_samples));

    std::vector<Candidate> bucket_male, bucket_female;
    bucket_male.reserve((std::size_t)max_samples * 2);
    bucket_female.reserve((std::size_t)max_samples * 2);

    cv::Mat frame;
    for (int idx = 0; idx < total; idx += step) {
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        if (!cap.read(frame) || frame.empty()) continue;
        for (auto& f : analyser.detect(frame)) {
            if (f.bbox.width < min_face_w) continue;
            if (f.embedding.empty())       continue;
            if      (f.gender == Gender::Male)   bucket_male  .push_back({idx, std::move(f)});
            else if (f.gender == Gender::Female) bucket_female.push_back({idx, std::move(f)});
        }
    }
    cap.release();

    auto pick_for = [](std::vector<Candidate>& bucket, std::vector<int>& consumed,
                       float thresh, SourceSpec& spec) -> bool {
        // Greedy clustering: for every unconsumed candidate, count how many
        // others (also unconsumed) are within `thresh` cosine, pick the one
        // with the largest cluster.
        const int n = (int)bucket.size();
        if (!n) return false;
        std::vector<bool> used(n, false);
        for (int i : consumed) if (i >= 0 && i < n) used[i] = true;

        int best = -1, best_pool = 0;
        std::vector<int> best_members;
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            std::vector<int> members{i};
            for (int j = 0; j < n; ++j) {
                if (i == j || used[j]) continue;
                if (cosine(bucket[i].face.embedding, bucket[j].face.embedding) >= thresh)
                    members.push_back(j);
            }
            if ((int)members.size() > best_pool) {
                best_pool = (int)members.size();
                best = i;
                best_members = std::move(members);
            }
        }
        if (best < 0) return false;

        spec.ref_emb   = bucket[best].face.embedding;
        spec.ref_frame = bucket[best].frame_idx;
        spec.ref_pool  = best_pool;
        spec.ref_votes = best_pool;
        for (int m : best_members) consumed.push_back(m);
        return true;
    };

    std::vector<int> consumed_male, consumed_female;
    for (auto& s : sources) {
        const float thresh = 0.30f;  // cluster threshold (tighter than match-time)
        bool ok = false;
        if (s.gender == Gender::Male)   ok = pick_for(bucket_male,   consumed_male,   thresh, s);
        else if (s.gender == Gender::Female) ok = pick_for(bucket_female, consumed_female, thresh, s);
        if (!ok) {
            // Fallback: use the source-image embedding so swap still works.
            s.ref_emb   = s.src_face.embedding;
            s.ref_frame = -1;
            s.ref_pool  = 0;
            s.ref_votes = 0;
        }
    }
}

}  // namespace faceswap
