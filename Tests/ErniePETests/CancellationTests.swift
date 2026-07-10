// CancellationTests.swift — ErniePE through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation); CAN-3 is the document of record for the checkpoint cadence. ErniePE is NOT
// long-run implied by the letter of the gate (llm is not a long-run capability; the declared
// footprints carry residentBytes only, no ≥ 2 GB transient) — but the autoregressive decode
// loop is a real per-token cadence over up to maxTokens (default 512), so the HONEST posture
// is the cadence the code actually has, declared instead of the exemption:
//   • the core's generate loop checks the injected `isCancelled` closure once per generated
//     token and bails with partial output (ErniePE/Model.swift `generate`, top of the token
//     loop; wired as `{ Task.isCancelled }` from ErniePEPackage.run for both the
//     promptEnhance and chat arms)
//   • the wrapper's post-generation `try Task.checkCancellation()` (both arms of
//     ErniePEPackage.run) rethrows the CancellationError unchanged — the qwen3-tts
//     non-throwing-core pattern.
// Note: ernie-pe vendors its OWN Mistral3 core (no mlx-swift-lm dependency), so the cadence
// is this repo's isCancelled seam, not MLXLMCommon's generate loop.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXErniePE

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config; construction is cheap (C13) — the default weight paths are only touched
        // by load(), and the entry checkpoint throws before validation, so this is offline-safe.
        let package = ErniePEPackage(configuration: ErniePEConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: LLMRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // Short-run envelope by the letter of the gate — but declare the real cadence anyway:
        // the autoregressive decode loop is a genuine multi-second run at 512 tokens.
        XCTAssertFalse(CancellationConformance.longRunImplied(by: ErniePEPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: ErniePEPackage.manifest,
            posture: .cadence([
                // Per generated token: `if isCancelled() { return (out, false) }` at the top of
                // the decode loop (ErniePE/Model.swift generate), rethrown unchanged by the
                // wrapper's post-generation checkpoint (ErniePEPackage.run, both arms).
                .init(phase: .generate, unit: .token),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
