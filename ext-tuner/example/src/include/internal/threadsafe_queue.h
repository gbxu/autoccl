#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T> class ThreadsafeQueue {
  public:
    ThreadsafeQueue() = default;

    void push(T &&value) {
      // std::lock_guard<std::mutex> lock(mutex_);
      pthread_mutex_lock(&this->mutex);
      queue_.push(std::move(value));
      // condition_.notify_one();
      pthread_cond_signal(&this->cond);
      pthread_mutex_unlock(&this->mutex);
    }

    void waitPop(T *value) {
      // std::unique_lock<std::mutex> lock(mutex_);
      pthread_mutex_lock(&this->mutex);
      // condition_.wait(lock, [this] { return !queue_.empty(); });
      while (queue_.empty()) {
        pthread_cond_wait(&this->cond, &this->mutex);
      }
      *value = std::move(queue_.front());
      queue_.pop();
      pthread_mutex_unlock(&this->mutex);
    }

    void merge(std::vector<T> *v) {
      for (auto &item : *v) {
        // std::lock_guard<std::mutex> lock(mutex_);
        pthread_mutex_lock(&this->mutex);
        queue_.push(std::move(item));
        // condition_.notify_one();
        pthread_cond_signal(&this->cond);
        pthread_mutex_unlock(&this->mutex);
      }
    }
    bool empty() {
      // std::lock_guard<std::mutex> lock(mutex_);
      pthread_mutex_lock(&this->mutex);
      bool flag = queue_.empty();  // NOLINT
      pthread_mutex_unlock(&this->mutex);
      return flag;
    }

    void wait(const bool &flag = false) {
      pthread_mutex_lock(&this->mutex);
      while (!flag && queue_.empty()) {
        pthread_cond_wait(&this->cond, &this->mutex);
      }
      pthread_mutex_unlock(&this->mutex);
    }

  private:
    std::queue<T> queue_;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    // mutable std::mutex mutex_;
    // std::condition_variable condition_;
};
