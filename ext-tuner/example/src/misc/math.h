#pragma once
#include <queue>
#include <utility>
#include <vector>
#include "src/include/datatype.h"
#include "src/include/internal/logging.h"

int32_t matchFilter(const Candidate &candidate, const CandidateFilter &filter) {
  auto mismatchPair =
      std::mismatch(candidate.begin(), candidate.end(), filter.begin());
  int32_t matchCount = std::distance(candidate.begin(), mismatchPair.first);
  return matchCount;
}

void filterCandidates(const std::vector<Candidate> &validCandidates,
                      const std::vector<CONFIG> &filter,
                      std::vector<Candidate> *filteredCandidates) {
  WARN(Logger::LogSubSys::ALL) << "It's very slow";
  if (filteredCandidates->size())
    CHECK(filter.size() < (*filteredCandidates)[0].size());
  for (const auto &candidate : validCandidates) {
    auto mismatchPair =
        std::mismatch(filter.begin(), filter.end(), candidate.begin());
    int32_t matchCount = std::distance(filter.begin(), mismatchPair.first);
    if (matchCount == filter.size()) {
      filteredCandidates->push_back(candidate);  // exactly same
    }
  }
  return;
}

// euclidean distance
double euclideanDistance(const Candidate &v1, const Candidate &v2,
                         const std::vector<ConfigRange> *configs = nullptr) {
  CHECK(v1.size() == v2.size()) << "v1=" << toDebugStr(v1) << ", v2=" << toDebugStr(v2);
  if (configs != nullptr) {
    CHECK(v1.size() == configs->size());
  }
  double distance = 0.0;
  for (size_t i = 0; i < v1.size(); ++i) {
    if (configs != nullptr) {
      if ((*configs)[i].begin == ConfigRange::EMPTY ||
          (*configs)[i].end == ConfigRange::EMPTY) {
        distance += std::pow(v1[i] - v2[i], 2) * (*configs)[i].weight;;
      } else {
        distance += std::pow(
            (v1[i] - v2[i]) * 1.0 / ((*configs)[i].begin - (*configs)[i].end), 2) * (*configs)[i].weight;
      }
    } else {
      distance += std::pow(v1[i] - v2[i], 2);
    }
  }
  return std::sqrt(distance);
}

// sort candidates
void sortCandidatesByProximity(std::vector<Candidate> &points,
                               const Candidate &referencePoint) {
  std::unordered_map<const Candidate *, double> distanceMap;
  for (const auto &point : points) {
    distanceMap[&point] = euclideanDistance(point, referencePoint);
  }

  std::sort(points.begin(), points.end(),
            [&distanceMap](const Candidate &a, const Candidate &b) {
              return distanceMap[&a] < distanceMap[&b];
            });
}

std::vector<Candidate>
findKClosestCandidates(const std::vector<Candidate> &points,
                       const Candidate &referencePoint, int k,
                       const std::vector<ConfigRange> *configs = nullptr) {
  // TODO: 不均等的距离
  // distance and candidate
  auto comp = [&referencePoint](const std::pair<double, const Candidate *> &a,
                                const std::pair<double, const Candidate *> &b) {
    return a.first < b.first;  // heap
  };
  std::priority_queue<std::pair<double, const Candidate *>,
                      std::vector<std::pair<double, const Candidate *>>,
                      decltype(comp)>
      pq(comp);

  // Traverse all points and maintain a maximum heap of size k 
  // O(n) * ( O(m) + O(logk))
  for (const auto &point : points) {
    // O(m) m=configs.size()
    double distance = euclideanDistance(point, referencePoint, configs);
    // O(logk)
    if (pq.size() < k) {
      pq.emplace(distance, &point);
    } else if (distance < pq.top().first) {
      pq.pop();
      pq.emplace(distance, &point);
    }
  }

  // extract the nearest k points from the heap
  // k * O(logk)
  std::vector<Candidate> closestCandidates;
  while (!pq.empty()) {
    closestCandidates.push_back(*pq.top().second);
    pq.pop();
  }

  // reverse
  // from near to far
  std::reverse(closestCandidates.begin(), closestCandidates.end());

  return closestCandidates;
}

std::vector<Candidate>
findKPrefixMatchCandidates(const std::vector<Candidate> &points,
                       const Candidate &referencePoint, int k) {
  auto countMatches = [](const Candidate &a,
                         const Candidate &b) {
    CHECK(a.size() == b.size());
    int matches = 0;
    for (int i = 0; i < a.size(); ++i) {
      if (a[i] == b[i]) {
        ++matches;
      }
    }
    return matches;
  };
  // Vector to store pairs of candidates and their match scores
  std::vector<std::pair<int, Candidate>> scores;
  // Calculate match scores
  for (const auto& point : points) {
      int score = countMatches(point, referencePoint);
      scores.push_back({score, point});
  }
  // Sort scores in descending order
  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
  });
  // Collect top k candidates
  std::vector<Candidate> result;
  for (int i = 0; i < k && i < scores.size(); ++i) {
      result.push_back(scores[i].second);
  }
  return result;
}

// calculates the dot product of two vectors
double dotProduct(const Candidate &a, const Candidate &b) {
  double dot = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

// Calculates the magnitude of a vector
double magnitude(const Candidate &a) {
  double mag = 0;
  for (double val : a) {
    mag += val * val;
  }
  return std::sqrt(mag);
}

// Calculates the angle between vector AB and the specified axis direction
double angleWithAxis(const Candidate &A, const Candidate &B, int32_t axis, bool positive) {
  Candidate AB(A.size()); // The vector from point A to point B
  for (size_t i = 0; i < A.size(); ++i) {
    AB[i] = B[i] - A[i];
  }
  Candidate axisVector(A.size(), 0); // Vector in the direction of the axis
  axisVector[axis] = positive ? 1 : -1;
  double dot = dotProduct(AB, axisVector);
  double magAB = magnitude(AB);
  double magAxis = magnitude(axisVector);
  // The smaller the return value, the closer it is to the specified direction.
  return std::acos(dot / (magAB * magAxis));
}

// Calculates the projection length of vector AB in the specified axis direction
double projectionLengthOnAxis(const Candidate &A, const Candidate &B, int32_t axis, bool positive) {
  Candidate AB(A.size()); // The vector from point A to point B
  for (size_t i = 0; i < A.size(); ++i) {
    AB[i] = B[i] - A[i];
  }
  Candidate axisVector(A.size(), 0); // Vector in the direction of the axis
  axisVector[axis] = positive ? 1 : -1; // Set the axis direction according to the positive parameter
  double dot = dotProduct(AB, axisVector);
  double magAxis = magnitude(axisVector);
  return dot / magAxis; // Projection length
}

// Find the first k points with the smallest angle with the reference point along the positive direction of the specified coordinate axis as the reference point
std::vector<Candidate>
findKCandidatesWithAxis(const std::vector<Candidate> &points,
                        const Candidate &referencePoint, int k, int32_t axis, bool positive, double step,
                        const std::vector<ConfigRange> *configs = nullptr) {
  // Use the max heap to store the first k points with the smallest angle and distance
  auto comp = [](const std::pair<std::pair<double, double>, const Candidate *> &a,
          const std::pair<std::pair<double, double>, const Candidate *> &b) {
    if (a.first.first != b.first.first) {
      return a.first.first < b.first.first;
    } else {
      return a.first.second < b.first.second;
    }
  };
  std::priority_queue<std::pair<std::pair<double, double>, const Candidate *>,
      std::vector<std::pair<std::pair<double, double>, const Candidate *>>,
      decltype(comp)>
  pq(comp);

  int32_t count = 0;
  double originalk = k;
  for (const auto &point : points) {
    double angle = angleWithAxis(referencePoint, point, axis, positive);
    if (angle >= M_PI/2) {
      continue;
    }
    count++;
  }
  k = std::max(k, static_cast<int>(step * count));

  for (const auto &point : points) {
    double angle = angleWithAxis(referencePoint, point, axis, positive);
    if (angle >= M_PI/2) {
      continue;
    }
    
    double length = projectionLengthOnAxis(referencePoint, point, axis, positive);
    // double distance = euclideanDistance(referencePoint, point, configs);

    // When the optimization space is smaller, the step is smaller, the k is smaller, and the maximum heap distance is smaller
    if (pq.size() < k) {
      pq.emplace(std::make_pair(length, angle), &point);
    } else if (comp(std::make_pair(std::make_pair(length, angle), &point), pq.top())) {
      pq.pop();
      pq.emplace(std::make_pair(length, angle), &point);
    }
  }

  std::vector<Candidate> closestCandidates;
  if (pq.size() > 0) {
    while (!pq.empty() && (closestCandidates.size() < originalk)) {
      closestCandidates.push_back(*pq.top().second);
      pq.pop();
    }
  }

  std::stringstream ss;
  for (auto c : closestCandidates) {
    ss << toDebugStr(c);
  }
  TRACE(Logger::LogSubSys::OPTIMIZER) << " curr point: " << toDebugStr(referencePoint)
    << " axis=" << axis << " positive=" << positive << " originalk=" << originalk << " step=" << step << "*count=" << count << " k=" << k << " --> " << ss.str();
  return closestCandidates;
}
