// json_parser.h — a minimal JSON parser, purpose-built for reading a `.tm`
// file's metadata block.
//
// STATUS: real, and deliberately NOT a general-purpose JSON library — no
// nlohmann/json or rapidjson is available to this sandbox (no network
// access to fetch one; see STATUS.md), so this implements exactly the
// JSON subset tinymind.runtime.format's metadata actually uses:
// objects, strings, numbers (int and float), booleans, null, and arrays
// of any of those (needed for quantization metadata's "scales": [...] —
// see tinymind/quantization/model_quantizer.py). Nesting is supported to
// arbitrary depth (objects/arrays may contain objects/arrays), since
// tinymind.model.config.ModelConfig.to_dict() nests one level
// (`{"model_config": {...}}`) and quantization metadata nests further
// (`{"per_tensor_metadata": {"<name>": {"bits": 8, ...}}}`).
//
// This is a small recursive-descent parser over a JSON value tree
// (JsonValue, a tagged union via std::variant), not a streaming/SAX
// parser — appropriate for a metadata block that's realistically at most
// a few KB (see tinymind.runtime.format's own _MAX_HEADER_LENGTH sanity
// bound, which this parser's caller — tm_reader.h — enforces before ever
// handing bytes to this parser).
#ifndef TINYMIND_JSON_PARSER_H
#define TINYMIND_JSON_PARSER_H

#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace tinymind {

class JsonParseError : public std::runtime_error {
public:
    explicit JsonParseError(const std::string& message) : std::runtime_error(message) {}
};

class JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::map<std::string, JsonValue>;

class JsonValue {
public:
    using Storage = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;

    JsonValue() : storage_(nullptr) {}
    JsonValue(std::nullptr_t) : storage_(nullptr) {}
    JsonValue(bool b) : storage_(b) {}
    JsonValue(double d) : storage_(d) {}
    JsonValue(const std::string& s) : storage_(s) {}
    JsonValue(const char* s) : storage_(std::string(s)) {}
    JsonValue(const JsonArray& a) : storage_(a) {}
    JsonValue(const JsonObject& o) : storage_(o) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(storage_); }
    bool is_object() const { return std::holds_alternative<JsonObject>(storage_); }
    bool is_array() const { return std::holds_alternative<JsonArray>(storage_); }
    bool is_string() const { return std::holds_alternative<std::string>(storage_); }
    bool is_number() const { return std::holds_alternative<double>(storage_); }
    bool is_bool() const { return std::holds_alternative<bool>(storage_); }

    const JsonObject& as_object() const {
        if (!is_object()) throw JsonParseError("JsonValue is not an object");
        return std::get<JsonObject>(storage_);
    }
    const JsonArray& as_array() const {
        if (!is_array()) throw JsonParseError("JsonValue is not an array");
        return std::get<JsonArray>(storage_);
    }
    const std::string& as_string() const {
        if (!is_string()) throw JsonParseError("JsonValue is not a string");
        return std::get<std::string>(storage_);
    }
    double as_number() const {
        if (!is_number()) throw JsonParseError("JsonValue is not a number");
        return std::get<double>(storage_);
    }
    int as_int() const { return static_cast<int>(as_number()); }
    bool as_bool() const {
        if (!is_bool()) throw JsonParseError("JsonValue is not a boolean");
        return std::get<bool>(storage_);
    }

    // Convenience accessor: obj.get("key") throws a clear error naming the
    // missing key, rather than the caller indexing a map and getting a
    // default-constructed (silently wrong) value back.
    const JsonValue& get(const std::string& key) const {
        const auto& obj = as_object();
        auto it = obj.find(key);
        if (it == obj.end()) {
            throw JsonParseError("missing required JSON key: " + key);
        }
        return it->second;
    }

    bool has(const std::string& key) const {
        return is_object() && as_object().count(key) > 0;
    }

private:
    Storage storage_;
};

namespace detail {

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text), pos_(0) {}

    JsonValue parse() {
        skip_whitespace();
        JsonValue value = parse_value();
        skip_whitespace();
        if (pos_ != text_.size()) {
            throw JsonParseError("unexpected trailing content after JSON value at position " +
                                 std::to_string(pos_));
        }
        return value;
    }

private:
    const std::string& text_;
    size_t pos_;

    char peek() {
        if (pos_ >= text_.size()) throw JsonParseError("unexpected end of JSON input");
        return text_[pos_];
    }

    char advance() {
        char c = peek();
        pos_++;
        return c;
    }

    void expect(char expected) {
        char c = advance();
        if (c != expected) {
            throw JsonParseError(std::string("expected '") + expected + "' but got '" + c +
                                 "' at position " + std::to_string(pos_ - 1));
        }
    }

    void skip_whitespace() {
        while (pos_ < text_.size() &&
              (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\n' || text_[pos_] == '\r')) {
            pos_++;
        }
    }

    JsonValue parse_value() {
        skip_whitespace();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return JsonValue(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number();
        throw JsonParseError(std::string("unexpected character '") + c + "' at position " +
                             std::to_string(pos_));
    }

    JsonValue parse_object() {
        expect('{');
        JsonObject obj;
        skip_whitespace();
        if (peek() == '}') {
            advance();
            return JsonValue(obj);
        }
        while (true) {
            skip_whitespace();
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            JsonValue value = parse_value();
            obj[key] = value;
            skip_whitespace();
            char next = advance();
            if (next == '}') break;
            if (next != ',') throw JsonParseError("expected ',' or '}' in object");
        }
        return JsonValue(obj);
    }

    JsonValue parse_array() {
        expect('[');
        JsonArray arr;
        skip_whitespace();
        if (peek() == ']') {
            advance();
            return JsonValue(arr);
        }
        while (true) {
            arr.push_back(parse_value());
            skip_whitespace();
            char next = advance();
            if (next == ']') break;
            if (next != ',') throw JsonParseError("expected ',' or ']' in array");
        }
        return JsonValue(arr);
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (true) {
            char c = advance();
            if (c == '"') break;
            if (c == '\\') {
                char escaped = advance();
                switch (escaped) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'n': result.push_back('\n'); break;
                    case 't': result.push_back('\t'); break;
                    case 'r': result.push_back('\r'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'u': {
                        // Minimal \uXXXX support: only the common ASCII
                        // range this metadata ever actually contains
                        // (identifiers, architecture tags); a full
                        // UTF-16-surrogate-pair decoder is out of scope
                        // for what a model-config/metadata block needs.
                        if (pos_ + 4 > text_.size()) throw JsonParseError("truncated \\u escape");
                        std::string hex = text_.substr(pos_, 4);
                        pos_ += 4;
                        int code_point = std::strtol(hex.c_str(), nullptr, 16);
                        result.push_back(static_cast<char>(code_point & 0xFF));
                        break;
                    }
                    default:
                        throw JsonParseError(std::string("invalid escape character '") + escaped + "'");
                }
            } else {
                result.push_back(c);
            }
        }
        return result;
    }

    JsonValue parse_bool() {
        if (text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return JsonValue(true);
        }
        if (text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return JsonValue(false);
        }
        throw JsonParseError("invalid literal at position " + std::to_string(pos_));
    }

    JsonValue parse_null() {
        if (text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return JsonValue(nullptr);
        }
        throw JsonParseError("invalid literal at position " + std::to_string(pos_));
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (peek() == '-') advance();
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) advance();
        if (pos_ < text_.size() && text_[pos_] == '.') {
            advance();
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) advance();
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            advance();
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) advance();
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) advance();
        }
        std::string number_text = text_.substr(start, pos_ - start);
        return JsonValue(std::strtod(number_text.c_str(), nullptr));
    }
};

}  // namespace detail

inline JsonValue parse_json(const std::string& text) {
    detail::JsonParser parser(text);
    return parser.parse();
}

}  // namespace tinymind

#endif  // TINYMIND_JSON_PARSER_H
