// Smoke: manifest + (gated) enhance-mode + chat-mode through the canonical llm
// surface. The decisive checks: enhance output is ENGLISH (template ownership
// works), rich (longer than input), and chat is coherent.
//
// Run: PE_SMOKE=1 swift test --filter PackageTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXErniePE

final class PackageTests: XCTestCase {
    func testManifest() {
        let m = ErniePEPackage.manifest
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].capability, .llm)
        XCTAssertTrue(m.surfaces[0].supportedModes.contains(.promptEnhance))
    }

    func testEnhanceAndChat() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["PE_SMOKE"] == "1", "PE_SMOKE=1")

        let package = ErniePEPackage(configuration: .init())
        try await package.load()

        // 1. promptEnhance mode — brief prompt + resolution -> rich English description.
        let enhanceStart = Date()
        let enhanceResponse = try await package.run(LLMRequest(
            prompt: "a lighthouse on a stormy coast",
            mode: .promptEnhance,
            metaData: ["width": .int(1024), "height": .int(1024)]))
        guard let enhanced = enhanceResponse as? LLMResponse else { return XCTFail("type") }
        print("[enhance] (\(Date().timeIntervalSince(enhanceStart))s)\n\(enhanced.text)\n")
        XCTAssertGreaterThan(enhanced.text.count, 200, "enhancement should be rich")
        let asciiRatio = Double(enhanced.text.unicodeScalars.filter { $0.isASCII }.count)
            / Double(max(enhanced.text.unicodeScalars.count, 1))
        XCTAssertGreaterThan(asciiRatio, 0.9, "enhancement should be English (template ownership)")

        // 2. General chat.
        let chatStart = Date()
        let chatResponse = try await package.run(LLMRequest(
            prompt: "In one sentence, why is the sky blue?"))
        guard let chat = chatResponse as? LLMResponse else { return XCTFail("type") }
        print("[chat] (\(Date().timeIntervalSince(chatStart))s)\n\(chat.text)\n")
        XCTAssertGreaterThan(chat.text.count, 20)

        await package.unload()
    }
}
