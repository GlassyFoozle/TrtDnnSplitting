/**
 * table4_runner — C++ low-overhead TRT chunked pipeline benchmark.
 *
 * Usage:
 *   ./build/table4_runner --config <path/to/critical_full.json> \
 *                         --repo   <path/to/trt_split_runtime_baseline> \
 *                         [--precision fp32|fp16] \
 *                         [--warmup 20] \
 *                         [--iters 200]
 *
 * Output JSON:
 *   results/table4/<model>_cpp_<variant>_<precision>.json
 */

#include "chunk_pipeline.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// ── CLI parsing ───────────────────────────────────────────────────────────────

struct Args {
    std::string config_path;
    std::string repo_root;
    std::string precision = "fp32";
    int         warmup    = 20;
    int         iters     = 200;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if ((s == "--config" || s == "-c") && i + 1 < argc) {
            a.config_path = argv[++i];
        } else if ((s == "--repo" || s == "-r") && i + 1 < argc) {
            a.repo_root = argv[++i];
        } else if (s == "--precision" && i + 1 < argc) {
            a.precision = argv[++i];
        } else if (s == "--warmup" && i + 1 < argc) {
            a.warmup = std::stoi(argv[++i]);
        } else if (s == "--iters" && i + 1 < argc) {
            a.iters = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << s << "\n";
            std::exit(1);
        }
    }
    if (a.config_path.empty()) {
        std::cerr << "Usage: table4_runner --config <path> --repo <path> "
                     "[--precision fp32|fp16] [--warmup N] [--iters N]\n";
        std::exit(1);
    }
    // Default repo_root = parent of parent of config (artifacts/split_configs/<model>/).
    if (a.repo_root.empty()) {
        a.repo_root = fs::path(a.config_path).parent_path().parent_path()
                                              .parent_path().parent_path().string();
        std::cerr << "[info] repo_root inferred as: " << a.repo_root << "\n";
    }
    return a;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Load config JSON.
    std::ifstream cfg_f(args.config_path);
    if (!cfg_f) {
        std::cerr << "Cannot open config: " << args.config_path << "\n";
        return 1;
    }
    nlohmann::json cfg;
    cfg_f >> cfg;

    std::string model_name = cfg.value("model", "unknown");
    std::string variant    = cfg.value("variant", "critical_full");
    int n_chunks = cfg.value("n_chunks", 0);

    std::cerr << "\n" << std::string(60, '=') << "\n"
              << "C++ table4_runner: " << model_name
              << " / " << variant << "  precision=" << args.precision
              << "  chunks=" << n_chunks << "\n"
              << "  warmup=" << args.warmup << "  iters=" << args.iters << "\n"
              << std::string(60, '=') << "\n";

    SilentLogger logger;

    // Build pipeline (loads all engines, allocates buffers).
    std::cerr << "\n[loading engines...]\n";
    ChunkPipeline pipeline(cfg, args.repo_root, args.precision, logger);
    std::cerr << "  all engines loaded.\n";

    // Run benchmark.
    std::cerr << "\n[benchmarking...]\n";
    PipelineResult res = pipeline.run(args.warmup, args.iters);

    // Print summary.
    std::cerr << "\n  full engine GPU mean = " << res.full_engine_gpu_mean_ms << " ms"
              << "  p99 = " << res.full_engine_gpu_p99_ms << " ms\n";
    std::cerr << "  chunked total GPU mean = " << res.total_chunked_gpu_mean_ms << " ms"
              << "  p99 = " << res.total_chunked_gpu_p99_ms << " ms\n";
    for (auto& c : res.chunks) {
        std::cerr << "  chunk" << c.id
                  << "  GPU mean=" << c.gpu_mean_ms << " ms"
                  << "  p99=" << c.gpu_p99_ms << " ms\n";
    }

    // Build output JSON.
    nlohmann::json out;
    out["model"]                        = model_name;
    out["variant"]                      = variant;
    out["precision"]                    = args.precision;
    out["n_chunks"]                     = n_chunks;
    out["n_iters"]                      = res.n_iters;
    out["full_engine_gpu_mean_ms"]      = res.full_engine_gpu_mean_ms;
    out["full_engine_gpu_p99_ms"]       = res.full_engine_gpu_p99_ms;
    out["total_chunked_gpu_mean_ms"]    = res.total_chunked_gpu_mean_ms;
    out["total_chunked_gpu_p99_ms"]     = res.total_chunked_gpu_p99_ms;

    nlohmann::json chunk_arr = nlohmann::json::array();
    for (auto& c : res.chunks) {
        nlohmann::json ch;
        ch["id"]          = c.id;
        ch["gpu_mean_ms"] = c.gpu_mean_ms;
        ch["gpu_p99_ms"]  = c.gpu_p99_ms;
        ch["cpu_mean_ms"] = c.cpu_mean_ms;
        ch["cpu_p99_ms"]  = c.cpu_p99_ms;
        chunk_arr.push_back(ch);
    }
    out["chunks"] = chunk_arr;

    // Write to results/table4/<model>_cpp_<variant>_<precision>.json
    fs::path out_dir = fs::path(args.repo_root) / "results" / "table4";
    fs::create_directories(out_dir);
    std::string out_name = model_name + "_cpp_" + variant + "_" + args.precision + ".json";
    fs::path out_path = out_dir / out_name;
    std::ofstream of(out_path);
    of << out.dump(2) << "\n";
    std::cerr << "\nSaved → " << out_path.string() << "\n";
    std::cout << out_path.string() << "\n";

    return 0;
}
