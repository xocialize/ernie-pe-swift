// MLXEngine `llm` package over the ErniePE core — the engine's first llm surface.
//
// Two behaviors, one surface (C4: mode rides the request, never a second tool):
//   - mode == .promptEnhance: messages' last user turn is a BRIEF image prompt;
//     metaData `width`/`height` carry the target resolution; the response text is a
//     rich visual description consumable by ANY textToImage backer (Lens, ERNIE-Turbo,
//     anything registered) — the package has no t2i dependency whatsoever.
//   - otherwise: general chat over the canonical messages (system + user/assistant).

import Foundation
import MLXToolKit
import MLXProfiling
import ErniePE
import Tokenizers

extension Mode {
    /// Expand a brief image prompt into a rich t2i-ready description.
    public static let promptEnhance: Mode = "promptEnhance"
}

/// Init-time configuration (C9): where the PE checkpoint lives.
public struct ErniePEConfiguration: PackageConfiguration, ModelStorable {
    /// Directory holding the PE weights (`pe/` of an ERNIE-Image-Turbo snapshot, or a
    /// standalone export) — config.json + safetensors.
    public var weightsPath: String
    /// Tokenizer directory (`pe_tokenizer/` — tokenizer.json etc.).
    public var tokenizerPath: String
    /// Converted 4-bit repo (weights.safetensors + config.json); preferred when set.
    public var quantizedPath: String?
    public var modelsRootDirectory: URL?

    public init(
        weightsPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo/pe",
        tokenizerPath: String =
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo/pe_tokenizer",
        quantizedPath: String? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.weightsPath = weightsPath
        self.tokenizerPath = tokenizerPath
        self.quantizedPath = quantizedPath
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case weightsPath, tokenizerPath, quantizedPath
    }
}

public enum ErniePEPackageError: Error, LocalizedError {
    case unreadableWeights(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableWeights(let p): return "PE weights not readable at \(p)."
        }
    }
}

@InferenceActor
public final class ErniePEPackage: ModelPackage {
    public typealias Configuration = ErniePEConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            provenance: Provenance(
                sourceRepo: "baidu/ERNIE-Image-Turbo", revision: "main", tier: 1),
            requirements: RequirementsManifest(
                // bf16 weights 7.2 GB (+ KV/activations); int4 ~2.4 GB.
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 9_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 4_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                LLMContract.descriptor(
                    name: "ernie-pe-3b",
                    summary: "Ministral-3B instruct LLM (the ERNIE Prompt Enhancer): general "
                        + "chat, plus promptEnhance mode that expands a brief image prompt + "
                        + "target resolution into a rich t2i-ready description for ANY "
                        + "registered textToImage model.",
                    modes: [.promptEnhance, .direct]
                )
            ]
        )
    }

    private let configuration: Configuration
    private var pipeline: ErniePEPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard pipeline == nil else { return }
        let model: ErniePEModel
        if let quantizedPath = configuration.quantizedPath {
            model = try ErniePEWeights.loadQuantized(directory: URL(fileURLWithPath: quantizedPath))
        } else {
            let dir = URL(fileURLWithPath: configuration.weightsPath)
            guard FileManager.default.fileExists(atPath: dir.path) else {
                throw ErniePEPackageError.unreadableWeights(dir.path)
            }
            model = try ErniePEWeights.loadFromPT(directory: dir)
        }
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: configuration.tokenizerPath))
        pipeline = ErniePEPipeline(model: model, tokenizer: tokenizer)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the core's autoregressive loop checks the
        // `isCancelled` closure once per generated token (ErniePE Model.generate) and bails
        // with partial output; the post-generation checkpoints below rethrow the
        // CancellationError unchanged. See CancellationTests for the cadence of record.
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        guard request.capability == .llm, let llm = request as? LLMRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let temperature = Float(llm.parameters.temperature ?? 0.6)
        let topP = Float(llm.parameters.topP ?? 0.95)
        let maxTokens = llm.parameters.maxTokens ?? 512
        let seed: UInt64 = 0

        // Shared env-gated instrument (MLX_PROFILE=1; zero overhead when unset). run()
        // always follows load(), so weight I/O never lands inside the run window.
        let prof = MLXProfiler.shared
        let quant = configuration.quantizedPath != nil ? "int4" : "bf16"

        if llm.mode == .promptEnhance {
            guard let prompt = llm.messages.last(where: { $0.role == .user })?.content else {
                throw PackageError.notLoaded  // no user prompt — nothing to enhance
            }
            let width = llm.metaData["width"].flatMap(Self.intValue) ?? 1024
            let height = llm.metaData["height"].flatMap(Self.intValue) ?? 1024
            let system = llm.messages.first(where: { $0.role == .system })?.content
            // A custom (e.g. Chinese) instruction disables the English-stabilizing prefix.
            let prefix = llm.metaData["responsePrefix"].flatMap(Self.stringValue)
                ?? (system == nil ? "A " : "")
            prof.beginRun("ernie-pe promptEnhance \(quant)")
            let text = pipeline.enhance(
                prompt: prompt, width: width, height: height, systemPrompt: system,
                responsePrefix: prefix,
                maxNewTokens: maxTokens, temperature: temperature, topP: topP, seed: seed,
                isCancelled: { Task.isCancelled })
            prof.endRun(denominators: ["token": Self.generatedTokenCount(prof)])
            try Task.checkCancellation()
            return LLMResponse(text: text, finishReason: .stop)
        }

        // General chat: canonical messages -> (system, alternating turns).
        let system = llm.messages.first(where: { $0.role == .system })?.content
        var turns: [(user: String, assistant: String?)] = []
        var pendingUser: String?
        for message in llm.messages {
            switch message.role {
            case .system: continue
            case .user:
                if let user = pendingUser { turns.append((user: user, assistant: nil)) }
                pendingUser = message.content
            case .assistant:
                turns.append((user: pendingUser ?? "", assistant: message.content))
                pendingUser = nil
            }
        }
        if let user = pendingUser { turns.append((user: user, assistant: nil)) }

        prof.beginRun("ernie-pe chat \(quant)")
        let (text, natural) = pipeline.chat(
            system: system, turns: turns, maxNewTokens: maxTokens,
            temperature: temperature, topP: topP, seed: seed,
            isCancelled: { Task.isCancelled })
        prof.endRun(denominators: ["token": Self.generatedTokenCount(prof)])
        try Task.checkCancellation()
        return LLMResponse(text: text, finishReason: natural ? .stop : .length)
    }

    /// Exact decode-token count the core marked as `llm/generated` — the pipeline returns
    /// text (not token ids), so the ms/token denominator rides the profiler's own rows.
    nonisolated static func generatedTokenCount(_ prof: MLXProfiler) -> Double {
        Double(prof.recorded.last(where: { $0.group == "llm" && $0.label == "generated" })?
            .index ?? 0)
    }

    nonisolated static func stringValue(_ value: MetaValue) -> String? {
        if case .string(let s) = value { return s }
        return nil
    }

    nonisolated static func intValue(_ value: MetaValue) -> Int? {
        switch value {
        case .int(let i): return i
        case .double(let d): return Int(d)
        case .string(let s): return Int(s)
        default: return nil
        }
    }
}

extension ErniePEPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(ErniePEPackage.self)
    }
}
