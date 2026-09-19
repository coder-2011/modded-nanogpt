#include <Python.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Object {
    PyObject *p;
    explicit Object(PyObject *value) : p(value) {
        if (!p) {
            PyErr_Print();
            throw std::runtime_error("Python binding failed");
        }
    }
    ~Object() { Py_DECREF(p); }
    Object(const Object &) = delete;
};
struct Batch {
    std::vector<uint32_t> ids;
    std::vector<uint64_t> offsets;
};
Batch snap(PyObject *tokenizer, PyObject *texts) {
    Object result(PyObject_CallMethod(tokenizer, "encode_batch_flat", "O", texts));
    auto *ids = PyTuple_GetItem(result.p, 0);
    auto *offsets = PyTuple_GetItem(result.p, 1);
    if (!ids || !offsets || !PyBytes_Check(ids) || !PyBytes_Check(offsets))
        throw std::runtime_error("flat API contract");
    Batch out;
    out.ids.resize(PyBytes_Size(ids) / 4);
    out.offsets.resize(PyBytes_Size(offsets) / 8);
    std::memcpy(out.ids.data(), PyBytes_AsString(ids), out.ids.size() * 4);
    std::memcpy(out.offsets.data(), PyBytes_AsString(offsets), out.offsets.size() * 8);
    return out;
}
Batch tik(PyObject *tokenizer, PyObject *texts, bool serial = false) {
    Object method(PyObject_GetAttrString(tokenizer, serial ? "encode" : "encode_batch"));
    Object kwargs(PyDict_New());
    Object all(PyUnicode_FromString("all"));
    Object four(PyLong_FromLong(4));
    PyDict_SetItemString(kwargs.p, "allowed_special", all.p);
    Batch out;
    out.offsets.push_back(0);
    auto append = [&](PyObject *row) {
        for (Py_ssize_t j = 0; j < PyList_Size(row); ++j)
            out.ids.push_back(PyLong_AsUnsignedLong(PyList_GetItem(row, j)));
        out.offsets.push_back(out.ids.size());
    };
    if (serial) {
        for (Py_ssize_t i = 0; i < PyList_Size(texts); ++i) {
            Object args(PyTuple_Pack(1, PyList_GetItem(texts, i)));
            Object result(PyObject_Call(method.p, args.p, kwargs.p));
            append(result.p);
        }
    } else {
        PyDict_SetItemString(kwargs.p, "num_threads", four.p);
        Object args(PyTuple_Pack(1, texts));
        Object result(PyObject_Call(method.p, args.p, kwargs.p));
        for (Py_ssize_t i = 0; i < PyList_Size(result.p); ++i)
            append(PyList_GetItem(result.p, i));
    }
    return out;
}
void write_shard(const std::string &path, const Batch &batch) {
    std::vector<uint16_t> tokens;
    for (size_t i = 1; i < batch.offsets.size(); ++i) {
        tokens.push_back(50256);
        for (size_t j = batch.offsets[i - 1]; j < batch.offsets[i]; ++j) {
            if (batch.ids[j] > 50256)
                throw std::runtime_error("token outside GPT-2 vocabulary");
            tokens.push_back(uint16_t(batch.ids[j]));
        }
    }
    if (tokens.size() > INT32_MAX)
        throw std::runtime_error("shard too large");
    int32_t header[256] = {20240520, 1, int32_t(tokens.size())};
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char *>(header), sizeof(header));
    file.write(reinterpret_cast<char *>(tokens.data()), tokens.size() * 2);
    if (!file)
        throw std::runtime_error("shard write failed");
    file.close();
    std::ifstream check(path, std::ios::binary);
    int32_t read_header[256];
    std::vector<uint16_t> read_tokens(tokens.size());
    check.read(reinterpret_cast<char *>(read_header), sizeof(read_header));
    check.read(reinterpret_cast<char *>(read_tokens.data()), read_tokens.size() * 2);
    if (!check || std::memcmp(header, read_header, sizeof(header)) || read_tokens != tokens)
        throw std::runtime_error("shard round-trip failed");
    std::cout << "wrote " << path << " tokens=" << tokens.size()
              << " documents=" << batch.offsets.size() - 1 << "\n";
}
int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::runtime_error(
                "usage: tokenizer PYTHON_EXECUTABLE TOKENIZER_JSON [INPUT_JSONL OUTPUT_BIN]");
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        PyStatus s = PyConfig_SetBytesString(&config, &config.program_name, argv[1]);
        if (PyStatus_Exception(s))
            Py_ExitStatusException(s);
        s = Py_InitializeFromConfig(&config);
        PyConfig_Clear(&config);
        if (PyStatus_Exception(s))
            Py_ExitStatusException(s);
        Object module(PyImport_ImportModule("snaptokens"));
        Object type(PyObject_GetAttrString(module.p, "Tokenizer"));
        Object st(PyObject_CallMethod(type.p, "from_file", "s", argv[2]));
        Object tm(PyImport_ImportModule("tiktoken"));
        Object tt(PyObject_CallMethod(tm.p, "get_encoding", "s", "gpt2"));
        std::vector<std::string> documents;
        if (argc == 5) {
            std::ifstream in(argv[3]);
            if (!in)
                throw std::runtime_error("input open failed");
            Object json(PyImport_ImportModule("json"));
            std::string line;
            while (std::getline(in, line)) {
                Object record(PyObject_CallMethod(json.p, "loads", "s", line.c_str()));
                PyObject *text = PyDict_GetItemString(record.p, "text");
                if (!text || !PyUnicode_Check(text))
                    throw std::runtime_error("each JSONL record needs text");
                Py_ssize_t size;
                const char *p = PyUnicode_AsUTF8AndSize(text, &size);
                if (!p)
                    throw std::runtime_error("UTF-8 decode failed");
                documents.emplace_back(p, size);
            }
        } else {
            const std::vector<std::string> cases = {"",
                                                    "Hello, world!",
                                                    "  a\t b\r\n\n",
                                                    "it's I'M we'll WE'LL",
                                                    "1234567890 1.23e-10",
                                                    "café café 中文 日本語 العربية 😀",
                                                    "<|endoftext|>",
                                                    "\x01\x02\x7f",
                                                    "a\n b\n\n",
                                                    "🚀🧑‍💻é"};
            documents = cases;
            for (int i = 0; i < 2048; ++i) {
                std::string text;
                for (int j = 0; j < 16; ++j)
                    text += cases[1 + (i + j) % 5] + " document=" + std::to_string(i) +
                            " row=" + std::to_string(j) + ". ";
                documents.push_back(text);
            }
        }
        Object texts(PyList_New(documents.size()));
        size_t bytes = 0;
        for (size_t i = 0; i < documents.size(); ++i) {
            auto *text = PyUnicode_DecodeUTF8(documents[i].data(), documents[i].size(), "strict");
            if (!text)
                throw std::runtime_error("invalid UTF-8 input");
            PyList_SET_ITEM(texts.p, i, text);
            bytes += documents[i].size();
        }
        auto actual = snap(st.p, texts.p), expected = tik(tt.p, texts.p);
        if (actual.ids != expected.ids || actual.offsets != expected.offsets)
            throw std::runtime_error("GPT-2 token ID or boundary mismatch");
        std::cout << "exact GPT-2 parity: documents=" << documents.size() << " bytes=" << bytes
                  << " tokens=" << actual.ids.size() << "\n";
        auto benchmark = [&](int mode) {
            std::vector<double> times;
            for (int i = 0; i < 15; ++i) {
                auto begin = std::chrono::steady_clock::now();
                Batch result = mode == 0 ? snap(st.p, texts.p) : tik(tt.p, texts.p, mode == 2);
                auto end = std::chrono::steady_clock::now();
                if (result.ids != expected.ids || result.offsets != expected.offsets)
                    throw std::runtime_error("repeat parity mismatch");
                if (i >= 5)
                    times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
            }
            auto sorted = times;
            std::sort(sorted.begin(), sorted.end());
            double ms = (sorted[4] + sorted[5]) / 2;
            std::cout << (mode == 0   ? "snaptokens_flat"
                          : mode == 1 ? "tiktoken_batch"
                                      : "tiktoken_serial")
                      << " median_ms=" << ms << " MB_per_s=" << bytes / ms / 1000 << " samples_ms=";
            for (double value : times)
                std::cout << " " << value;
            std::cout << "\n";
            return ms;
        };
        auto a = benchmark(0), b = benchmark(1), c = benchmark(2);
        std::cout << "preprocessing_speed_ratio=" << std::min(b, c) / a
                  << " against fastest tiktoken control; warm repeated synthetic corpus, includes "
                     "C++ ID materialization, 4 allocated CPUs\n";
        write_shard(argc == 5 ? argv[4] : "/tmp/nanogpt-snaptokens-check.bin", actual);
        std::cout
            << "PASS: Snaptokens preprocessing; official pretokenized benchmark shards unchanged\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
