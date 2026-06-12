// t2v suitability probe: same brief prompt, (a) default t2i enhancement vs
// (b) a video-enhancement system prompt (motion / camera / temporal arc).
// Run: PE_T2V=1 swift test --filter VideoEnhanceTests

import Foundation
import MLXToolKit
import XCTest

@testable import MLXErniePE

final class VideoEnhanceTests: XCTestCase {
    static let videoSystemPrompt =
        "You are a professional text-to-video prompt enhancement assistant. You will "
        + "receive a brief scene description and the target resolution. Expand it into a "
        + "single richly detailed VIDEO prompt IN ENGLISH for a text-to-video model: "
        + "describe the subject and setting, the MOTION of subjects and environment over "
        + "the clip, the CAMERA work (e.g. slow dolly-in, pan, static tripod), and how the "
        + "scene evolves from the first frame to the last. Keep it one continuous shot. "
        + "Output only the enhanced prompt, without any explanation or prefix."

    func testVideoVsImageEnhance() async throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["PE_T2V"] == "1", "PE_T2V=1")
        let package = ErniePEPackage(configuration: .init(
            quantizedPath: "/Volumes/DEV_VOL1/VideoResearch/ernie-image-models/ERNIE-PE-3B-4bit"))
        try await package.load()

        let brief = "a sailboat at sunset"
        let image = try await package.run(LLMRequest(
            prompt: brief, mode: .promptEnhance,
            metaData: ["width": .int(1024), "height": .int(576)])) as! LLMResponse
        print("[t2i enhance]\n\(image.text)\n")

        let video = try await package.run(LLMRequest(
            messages: [
                ChatMessage(role: .system, content: Self.videoSystemPrompt),
                ChatMessage(role: .user, content: brief),
            ],
            mode: .promptEnhance,
            metaData: ["width": .int(1024), "height": .int(576)])) as! LLMResponse
        print("[t2v enhance]\n\(video.text)\n")

        // Motion vocabulary should appear in the video variant.
        let motionWords = ["camera", "dolly", "pan", "glides", "moves", "drifts", "slowly",
                           "tracking", "zoom", "begins", "as the", "then"]
        let hits = motionWords.filter { video.text.lowercased().contains($0) }
        print("[motion-vocabulary hits] \(hits)")
        XCTAssertGreaterThanOrEqual(hits.count, 3, "video enhancement should describe motion/camera")
        await package.unload()
    }
}
