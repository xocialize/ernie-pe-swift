// ERNIE Prompt Enhancer — Ministral-3B-Instruct fine-tune (pe/ in ERNIE-Image-Turbo).
//
// The backbone is byte-for-byte the architecture parity-locked in ernie-image-swift's
// text encoder (26L / hidden 3072 / GQA 32q-8kv / head_dim 128 / ffn 9216 SwiGLU /
// RMSNorm eps 1e-5 / no biases / vocab 131072 / **YaRN rope** factor 16, theta 1e6 —
// the same resolved inv_freq we gated exactly vs the HF dump), vendored here so this
// package stands alone. Deltas for GENERATION: KV cache, the final norm IS applied,
// and the lm_head (shipped in pe/ despite tie_word_embeddings=true — the file wins).
//
// Generation defaults follow the reference pipeline: temperature 0.6, top_p 0.95,
// eos 2, bos 1, pad 11.

import Foundation
import MLX
import MLXFast
import MLXNN

public enum ErniePEError: Error, CustomStringConvertible {
    case loading(String)
    case generation(String)

    public var description: String {
        switch self {
        case .loading(let m): return "ErniePE loading error: \(m)"
        case .generation(let m): return "ErniePE generation error: \(m)"
        }
    }
}

// MARK: - YaRN rope (identical to the ernie-image-swift gated implementation)

public enum PEYarnRope {
    public static let theta: Float = 1_000_000
    public static let headDim = 128
    public static let factor: Float = 16
    public static let betaFast: Float = 32
    public static let betaSlow: Float = 1
    public static let originalMaxPositions: Float = 16384

    public static func invFreq() -> [Float] {
        let half = headDim / 2
        func correctionDim(_ numRotations: Float) -> Float {
            Float(headDim) * log(originalMaxPositions / (numRotations * 2 * .pi))
                / (2 * log(theta))
        }
        let lowC = max(floor(correctionDim(betaFast)), 0)
        let highC = min(ceil(correctionDim(betaSlow)), Float(half - 1))
        var out = [Float](repeating: 0, count: half)
        for i in 0..<half {
            let posFreq = pow(theta, Float(2 * i) / Float(headDim))
            var ramp = (Float(i) - lowC) / max(highC - lowC, 0.001)
            ramp = min(max(ramp, 0), 1)
            let mask = 1 - ramp
            out[i] = (1.0 / (factor * posFreq)) * (1 - mask) + (1.0 / posFreq) * mask
        }
        return out
    }

    /// cos/sin for absolute positions [offset, offset+length).
    static func cosSin(offset: Int, length: Int) -> (MLXArray, MLXArray) {
        let inv = MLXArray(invFreq())
        let pos = MLXArray(offset..<(offset + length)).asType(.float32)
        let freqs = pos[0..., .newAxis] * inv[.newAxis, 0...]
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }

    static func apply(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
        let d = x.dim(-1)
        let x1 = x[.ellipsis, ..<(d / 2)]
        let x2 = x[.ellipsis, (d / 2)...]
        let rotated = concatenated([-x2, x1], axis: -1)
        let c = cosT[.newAxis, .newAxis].asType(x.dtype)
        let s = sinT[.newAxis, .newAxis].asType(x.dtype)
        return x * c + rotated * s
    }
}

// MARK: - KV cache (growing concat; one per layer)

public final class PEKVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int { keys?.dim(2) ?? 0 }

    func update(keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            self.keys = k
            self.values = v
        }
        return (self.keys!, self.values!)
    }
}

// MARK: - Layers

final class PEAttention: Module {
    let heads = 32
    let kvHeads = 8
    let headDim = 128

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(hidden: Int) {
        self._wq.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: PEKVCache
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = wq(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = PEYarnRope.apply(q, cos: cos, sin: sin)
        k = PEYarnRope.apply(k, cos: cos, sin: sin)
        (k, v) = cache.update(keys: k, values: v)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: mask)
        return wo(out.transposed(0, 2, 1, 3).reshaped(b, l, heads * headDim))
    }
}

final class PEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, ffn: Int) {
        self._gate.wrappedValue = Linear(hidden, ffn, bias: false)
        self._up.wrappedValue = Linear(hidden, ffn, bias: false)
        self._down.wrappedValue = Linear(ffn, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class PELayer: Module {
    @ModuleInfo(key: "self_attn") var attention: PEAttention
    @ModuleInfo(key: "mlp") var mlp: PEMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(hidden: Int, ffn: Int, eps: Float) {
        self._attention.wrappedValue = PEAttention(hidden: hidden)
        self._mlp.wrappedValue = PEMLP(hidden: hidden, ffn: ffn)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._postNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: PEKVCache
    ) -> MLXArray {
        let h = x + attention(inputNorm(x), cos: cos, sin: sin, mask: mask, cache: cache)
        return h + mlp(postNorm(h))
    }
}

// MARK: - Model

public final class ErniePEModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [PELayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public static let bosToken = 1
    public static let eosToken = 2
    public static let padToken = 11

    public init(hidden: Int = 3072, ffn: Int = 9216, numLayers: Int = 26,
                vocab: Int = 131_072, eps: Float = 1e-5) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            PELayer(hidden: hidden, ffn: ffn, eps: eps)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._lmHead.wrappedValue = Linear(hidden, vocab, bias: false)
        super.init()
    }

    /// One forward over `ids` continuing from the caches; returns last-position logits.
    func step(_ ids: MLXArray, caches: [PEKVCache]) -> MLXArray {
        var h = embedTokens(ids)
        let offset = caches[0].offset
        let (cosT, sinT) = PEYarnRope.cosSin(offset: offset, length: ids.dim(1))
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = ids.dim(1) > 1 ? .causal : .none
        for (i, layer) in layers.enumerated() {
            h = layer(h, cos: cosT, sin: sinT, mask: mask, cache: caches[i])
        }
        return lmHead(norm(h[0..., -1, 0...][0..., .newAxis, 0...])).squeezed(axis: 1)
    }

    /// Sampled generation (reference defaults: temp 0.6, top-p 0.95).
    public func generate(
        promptIds: [Int],
        maxNewTokens: Int = 512,
        temperature: Float = 0.6,
        topP: Float = 0.95,
        seed: UInt64 = 0,
        isCancelled: () -> Bool = { false }
    ) -> (tokens: [Int], finishedNaturally: Bool) {
        let caches = (0..<layers.count).map { _ in PEKVCache() }
        var logits = step(
            MLXArray(promptIds.map { Int32($0) }).expandedDimensions(axis: 0), caches: caches)
        MLXRandom.seed(seed)
        var out: [Int] = []
        for _ in 0..<maxNewTokens {
            if isCancelled() { return (out, false) }
            let next = Self.sample(logits: logits, temperature: temperature, topP: topP)
            if next == Self.eosToken { return (out, true) }
            out.append(next)
            logits = step(MLXArray([Int32(next)]).expandedDimensions(axis: 0), caches: caches)
        }
        return (out, false)
    }

    /// Temperature + nucleus (top-p) sampling; greedy when temperature == 0.
    static func sample(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        if temperature <= 0 {
            return argMax(logits, axis: -1).item(Int.self)
        }
        let probs = softmax(logits[0].asType(.float32) / temperature, axis: -1)
        if topP < 1 {
            // Nucleus: keep the smallest prefix of descending probs with cumsum >= topP.
            let order = argSort(probs, axis: -1)  // ascending
            let sortedDesc = takeAlong(probs, order, axis: -1)[.stride(by: -1)]
            let cumulative = cumsum(sortedDesc, axis: -1)
            let cutoffIndex = sum(cumulative .< topP).item(Int.self)  // count strictly below
            let kept = min(cutoffIndex + 1, probs.dim(0))
            let threshold = sortedDesc[kept - 1].item(Float.self)
            let filtered = MLX.which(probs .>= threshold, probs, MLXArray(Float(0)))
            let renorm = filtered / sum(filtered)
            let sampled = MLXRandom.categorical(log(renorm + 1e-12)[.newAxis, 0...])
            return sampled.item(Int.self)
        }
        let sampled = MLXRandom.categorical(log(probs + 1e-12)[.newAxis, 0...])
        return sampled.item(Int.self)
    }
}
