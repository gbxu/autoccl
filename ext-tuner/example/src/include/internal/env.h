#pragma once
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

class Environment {
  public:
    static Environment *get() {
      static std::shared_ptr<Environment> instance(new Environment());
      return instance.get();
    }

    template <typename T> T find(const char *key, const T defaultValue) {
      const char *value = this->find(key);
      if (value == nullptr) {
        return defaultValue;
      } else {
        return T(value);
      }
    }

    int32_t find(const char *key, const int32_t defaultValue) {
      const char *value = this->find(key);
      if (value == nullptr) {
        return defaultValue;
      } else {
        return std::stoi(value);
      }
    }

    const char *find(const char *key) {
      if (kvs.count(std::string(key)) > 0) {
        return kvs[std::string(key)].c_str();
      } else {
        const char *value = getenv(key);
        if (value == nullptr || strlen(value) == 0)
          return nullptr;
        return value;
      }
    }

    void set(const std::string &key, const std::string &value) {
      kvs[key] = value;
    }

  private:
    std::unordered_map<std::string, std::string> kvs;
};

#if 0
int32_t main() {
  // Test the Environment class
  Environment& env = Environment::get();

  // Test find() with string value
  std::string strValue = env.find("STRING_KEY", "default_string");
  std::cout << "String value: " << strValue << std::endl;

  // Test find() with int32_t value
  int32_t intValue = env.find("INT_KEY", 123);
  std::cout << "Int value: " << intValue << std::endl;

  // Test set() and find() with newly set value
  env.set("NEW_KEY", "new_value");
  std::string newValue = env.find("NEW_KEY", "default_new_value");
  std::cout << "New value: " << newValue << std::endl;

  return 0;
}
#endif
