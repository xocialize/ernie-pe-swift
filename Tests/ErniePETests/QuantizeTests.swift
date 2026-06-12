// 4-bit conversion + quantized smoke.
// Convert: PE_CONVERT=1 swift test --filter QuantizeTests/testConvert
// Smoke:   PE_Q4_SMOKE=1 swift test --filter QuantizeTests/testQuantizedSmoke

import Foundation
import MLX
import MLXToolKit
import Tokenizers
import XCTest

@testable import ErniePE
@testable import MLXErniePE

final class QuantizeTests: XCTestCase {
    static let q4Dir = URL(
        fileURLWithPath:
            "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-PE-3B-4bit")

    func testConvert() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["PE_CONVERT"] == "1", "PE_CONVERT=1")
        let model = try ErniePEWeights.loadFromPT(
            directory: URL(fileURLWithPath: ErniePEConfiguration().weightsPath))
        try ErniePEWeights.saveQuantized(model: model, directory: Self.q4Dir)
        print("[converted] ERNIE-PE-3B-4bit")
    }

    func testQuantizedSmoke() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["PE_Q4_SMOKE"] == "1", "PE_Q4_SMOKE=1")
        let package = ErniePEPackage(
            configuration: .init(quantizedPath: Self.q4Dir.path))
        try await package.load()
        print("[q4] resident \(GPU.activeMemory / 1_000_000) MB")
        let response = try await package.run(LLMRequest(
            prompt: "a lighthouse on a stormy coast",
            mode: .promptEnhance,
            metaData: ["width": .int(1024), "height": .int(1024)]))
        guard let enhanced = response as? LLMResponse else { return XCTFail("type") }
        print("[q4 enhance]\n\(enhanced.text)")
        XCTAssertGreaterThan(enhanced.text.count, 200)
        await package.unload()
    }
}
