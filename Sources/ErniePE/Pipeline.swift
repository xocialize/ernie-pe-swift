// Loading, templates, and the two behaviors (prompt enhancement / general chat).
//
// The PE checkpoint's own chat_template.jinja HARDCODES a Chinese system prompt and
// drops system messages (validated 2026-06-12: English in -> Chinese out; in-prompt
// hints do NOT steer). We therefore OWN the template: standard Mistral control tokens
// with our system prompt. The English enhancement prompt below is a translation of
// the baked-in one, validated to produce full-quality English enhancements.

import Foundation
import MLX
import MLXNN
import Tokenizers

public enum ErniePEWeights {

    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw ErniePEError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    /// pe/ keys: `model.*` (backbone) + `lm_head.weight`.
    static func sanitize(_ k: String) -> String {
        k.hasPrefix("model.") ? String(k.dropFirst("model.".count)) : k
    }

    public static func loadFromPT(directory: URL, dtype: DType = .bfloat16) throws -> ErniePEModel {
        let model = ErniePEModel()
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            weights[sanitize(k)] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "ErniePE")
        return model
    }

    /// Convert to 4-bit and save (the ernie-image-swift recipe, vendored).
    public static func saveQuantized(
        model: ErniePEModel, directory: URL, groupSize: Int = 64, bits: Int = 4
    ) throws {
        quantize(model: model, groupSize: groupSize, bits: bits) { _, module in
            module is Linear
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let flat = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(arrays: flat, url: directory.appendingPathComponent("weights.safetensors"))
        let config: [String: Any] = ["quantization": ["group_size": groupSize, "bits": bits]]
        try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted])
            .write(to: directory.appendingPathComponent("config.json"))
    }

    public static func loadQuantized(directory: URL) throws -> ErniePEModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let cfg = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
              let q = cfg["quantization"] as? [String: Any],
              let groupSize = q["group_size"] as? Int, let bits = q["bits"] as? Int
        else { throw ErniePEError.loading("unreadable quantization config") }
        let model = ErniePEModel()
        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("weights.safetensors"))
        quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
            module is Linear && weights["\(path).scales"] != nil
        }
        try verifyAndLoad(model: model, weights: weights, label: "ErniePE(4bit)")
        return model
    }

    static func verifyAndLoad(model: Module, weights: [String: MLXArray], label: String) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw ErniePEError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                    + missing.prefix(4).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw ErniePEError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                    + unused.prefix(4).joined(separator: ", "))
        }
        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
    }
}

/// Mistral-instruct templating + the two behaviors.
public final class ErniePEPipeline {
    public let model: ErniePEModel
    public let tokenizer: any Tokenizers.Tokenizer

    /// English translation of the checkpoint's baked-in (Chinese) enhancement
    /// instruction — validated to produce full-quality English enhancements.
    public static let enhanceSystemPrompt =
        "You are a professional text-to-image prompt enhancement assistant. You will "
        + "receive a brief image description and the target generation resolution. Expand "
        + "it into a single richly detailed visual description IN ENGLISH to help a "
        + "text-to-image model generate a high-quality image. Output only the enhanced "
        + "description, without any explanation or prefix."

    public static let defaultChatSystemPrompt = "You are a helpful assistant."

    public init(model: ErniePEModel, tokenizer: any Tokenizers.Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Standard Mistral control-token template. `<s>` rides the string; encode with
    /// addSpecialTokens=false so special tokens match atomically and BOS isn't doubled.
    static func render(system: String, turns: [(user: String, assistant: String?)]) -> String {
        var s = "<s>[SYSTEM_PROMPT]\(system)[/SYSTEM_PROMPT]"
        for turn in turns {
            s += "[INST]\(turn.user)[/INST]"
            if let assistant = turn.assistant {
                s += "\(assistant)</s>"
            }
        }
        return s
    }

    func generate(
        text: String, maxNewTokens: Int, temperature: Float, topP: Float, seed: UInt64,
        isCancelled: () -> Bool = { false }
    ) -> (text: String, finishedNaturally: Bool) {
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let (tokens, natural) = model.generate(
            promptIds: ids, maxNewTokens: maxNewTokens, temperature: temperature,
            topP: topP, seed: seed, isCancelled: isCancelled)
        return (tokenizer.decode(tokens: tokens).trimmingCharacters(in: .whitespacesAndNewlines),
                natural)
    }

    /// Prompt enhancement for ANY t2i model: brief prompt + target resolution -> rich
    /// description. `systemPrompt` overrides the default English instruction (e.g. the
    /// original Chinese one for Chinese-prompt models).
    public func enhance(
        prompt: String, width: Int = 1024, height: Int = 1024,
        systemPrompt: String? = nil, maxNewTokens: Int = 512,
        temperature: Float = 0.6, topP: Float = 0.95, seed: UInt64 = 0,
        isCancelled: () -> Bool = { false }
    ) -> String {
        let payload: [String: Any] = ["prompt": prompt, "width": width, "height": height]
        let json = String(
            data: try! JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]),
            encoding: .utf8)!
        let text = Self.render(
            system: systemPrompt ?? Self.enhanceSystemPrompt, turns: [(user: json, assistant: nil)])
        let (out, _) = generate(
            text: text, maxNewTokens: maxNewTokens, temperature: temperature, topP: topP,
            seed: seed, isCancelled: isCancelled)
        return out
    }

    /// General chat over (system?, alternating user/assistant) turns.
    public func chat(
        system: String?, turns: [(user: String, assistant: String?)],
        maxNewTokens: Int = 512, temperature: Float = 0.6, topP: Float = 0.95,
        seed: UInt64 = 0, isCancelled: () -> Bool = { false }
    ) -> (text: String, finishedNaturally: Bool) {
        let text = Self.render(system: system ?? Self.defaultChatSystemPrompt, turns: turns)
        return generate(
            text: text, maxNewTokens: maxNewTokens, temperature: temperature, topP: topP,
            seed: seed, isCancelled: isCancelled)
    }
}
