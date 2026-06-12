// Prefix-forcing probe: does seeding the assistant turn stabilize (a) English for
// t2i and (b) motion/camera language for t2v?
// Run: PE_PREFIX=1 swift test --filter PrefixProbeTests

import Foundation
import Tokenizers
import XCTest

@testable import ErniePE
@testable import MLXErniePE

final class PrefixProbeTests: XCTestCase {
    func testPrefixForcing() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["PE_PREFIX"] == "1", "PE_PREFIX=1")
        let model = try ErniePEWeights.loadQuantized(
            directory: URL(fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-PE-3B-4bit"))
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath:
                "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-Image-Turbo/pe_tokenizer"))
        let pipeline = ErniePEPipeline(model: model, tokenizer: tokenizer)

        func probe(_ label: String, system: String, user: String, prefix: String) {
            let text = ErniePEPipeline.render(system: system, turns: [(user: user, assistant: nil)])
                + prefix
            let (out, _) = pipeline.generate(
                text: text, maxNewTokens: 400, temperature: 0.6, topP: 0.95, seed: 1)
            print("[\(label)]\n\(prefix)\(out)\n")
        }

        // (a) t2i English, prefix-forced
        probe("t2i + prefix",
              system: ErniePEPipeline.enhanceSystemPrompt,
              user: #"{"height":576,"prompt":"a sailboat at sunset","width":1024}"#,
              prefix: "A ")

        // (b) t2v, motion-forcing prefix
        probe("t2v + prefix",
              system: VideoEnhanceTests.videoSystemPrompt,
              user: #"{"height":576,"prompt":"a sailboat at sunset","width":1024}"#,
              prefix: "A single continuous shot: the camera ")
    }
}
