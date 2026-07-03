// swift-tools-version: 6.2
// ernie-pe-swift — the ERNIE Prompt Enhancer (Ministral-3B-Instruct fine-tune,
// Apache-2.0, ships inside baidu/ERNIE-Image-Turbo's pe/ dir) as a STANDALONE
// engine `llm` package. Deliberately independent of every t2i package: prompt
// enhancement is an LLM MODE (C4) whose text output feeds ANY textToImage backer
// (Lens, ERNIE-Turbo, future models) — or no t2i at all (it is a competent
// general 3B chat model; validated 2026-06-12).
//
// Backbone = the same Mistral3 architecture parity-locked in ernie-image-swift
// (26L/3072/GQA 32-8/head 128, YaRN rope factor 16) — vendored here, NOT a
// package dependency, so this repo stands alone.

import PackageDescription

let package = Package(
    name: "ErniePE",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "ErniePE", targets: ["ErniePE"]),
        .library(name: "MLXErniePE", targets: ["MLXErniePE"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/mlx-profiling.git", from: "0.1.0"),
    ],
    targets: [
        .target(
            name: "ErniePE",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            path: "Sources/ErniePE"
        ),
        .target(
            name: "MLXErniePE",
            dependencies: [
                "ErniePE",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXProfiling", package: "mlx-profiling"),
            ],
            path: "Sources/MLXErniePE"
        ),
        .testTarget(
            name: "ErniePETests",
            dependencies: ["ErniePE", "MLXErniePE"],
            path: "Tests/ErniePETests"
        ),
    ]
)
